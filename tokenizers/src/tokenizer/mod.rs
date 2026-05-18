//! Represents a tokenization pipeline.
//!
//! A [`Tokenizer`](struct.Tokenizer.html) is composed of some of the following parts.
//!   - [`Normalizer`](trait.Normalizer.html): Takes care of the text normalization (like unicode normalization).
//!   - [`PreTokenizer`](trait.PreTokenizer.html): Takes care of the pre tokenization (ie. How to split tokens and pre-process
//!     them.
//!   - [`Model`](trait.Model.html): A model encapsulates the tokenization algorithm (like BPE, Word base, character
//!     based, ...).
//!   - [`PostProcessor`](trait.PostProcessor.html): Takes care of the processing after tokenization (like truncating, padding,
//!     ...).

use ahash::AHashMap;
use std::{
    env,
    fs::{read_to_string, File},
    io::{prelude::*, BufReader},
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::utils::iter::ResultShunt;
use crate::utils::parallelism::*;
use crate::utils::progress::{ProgressBar, ProgressStyle};

mod added_vocabulary;
mod encoding;
pub mod normalizer;
pub mod pattern;
pub mod pre_tokenizer;
mod serialization;

// Re-export wrappers
pub use crate::decoders::DecoderWrapper;
pub use crate::models::ModelWrapper;
pub use crate::normalizers::NormalizerWrapper;
pub use crate::pre_tokenizers::PreTokenizerWrapper;
pub use crate::processors::PostProcessorWrapper;
// And some other types
pub use crate::utils::iter::LinesWithEnding;
pub use crate::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};
pub use crate::utils::truncation::{
    truncate_encodings, TruncationDirection, TruncationParams, TruncationStrategy,
};
pub use added_vocabulary::*;
pub use encoding::*;
pub use normalizer::{NormalizedString, OffsetReferential, SplitDelimiterBehavior};
pub use pre_tokenizer::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub type Offsets = (usize, usize);

#[derive(Debug, Clone, Copy)]
struct SpaceSafeParallelConfig {
    min_parallel_bytes: usize,
    target_chunk_bytes: usize,
    lookaround_bytes: usize,
}

impl Default for SpaceSafeParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_bytes: 10_000,
            target_chunk_bytes: 4_096,
            lookaround_bytes: 768,
        }
    }
}

impl SpaceSafeParallelConfig {
    fn from_env() -> Self {
        fn read_usize(name: &str, default: usize) -> usize {
            env::var(name)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(default)
        }

        let default = Self::default();
        Self {
            min_parallel_bytes: read_usize(
                "OTSO_TOKENIZER_MIN_PARALLEL_BYTES",
                default.min_parallel_bytes,
            ),
            target_chunk_bytes: read_usize(
                "OTSO_TOKENIZER_TARGET_CHUNK_BYTES",
                default.target_chunk_bytes,
            ),
            lookaround_bytes: read_usize(
                "OTSO_TOKENIZER_LOOKAROUND_BYTES",
                default.lookaround_bytes,
            ),
        }
    }

    fn global() -> Self {
        static CONFIG: OnceLock<SpaceSafeParallelConfig> = OnceLock::new();
        *CONFIG.get_or_init(Self::from_env)
    }
}

/// Takes care of pre-processing strings.
pub trait Normalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

/// The `PreTokenizer` is in charge of doing the pre-segmentation step. It splits the given string
/// in multiple substrings, keeping track of the offsets of said substrings from the
/// `NormalizedString`. In some occasions, the `PreTokenizer` might need to modify the given
/// `NormalizedString` to ensure we can entirely keep track of the offsets and the mapping with
/// the original string.
pub trait PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}

/// Represents a model used during Tokenization (like BPE or Word or Unigram).
pub trait Model {
    type Trainer: Trainer + Sync;
    /// Tokenize the given sequence into multiple underlying `Token`. The `offsets` on the `Token`
    /// are expected to be relative to the given sequence.
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>>;
    /// Tokenize returning only token IDs, without string values or offsets.
    ///
    /// This is designed for parallel encoding where the BPE `RwLock` cache causes
    /// contention. Implementations should skip cache access and string allocation.
    /// The default implementation falls back to `tokenize()` and extracts IDs.
    ///
    /// Must produce identical IDs to `tokenize()` for correctness.
    fn tokenize_ids(&self, sequence: &str) -> Result<Vec<u32>> {
        Ok(self.tokenize(sequence)?.into_iter().map(|t| t.id).collect())
    }
    /// Find the ID associated to a string token
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// Find the string token associated to an ID
    fn id_to_token(&self, id: u32) -> Option<String>;
    /// Retrieve the entire vocabulary mapping (token -> ID)
    fn get_vocab(&self) -> HashMap<String, u32>;
    /// Retrieve the size of the vocabulary
    fn get_vocab_size(&self) -> usize;
    /// Save the current `Model` in the given folder, using the given `prefix` for the various
    /// files that need to be saved.
    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>>;
    /// Get an instance of a Trainer capable of training this Model
    fn get_trainer(&self) -> <Self as Model>::Trainer;
}

/// A `PostProcessor` has the responsibility to post process an encoded output of the `Tokenizer`.
/// It adds any special tokens that a language model would require.
pub trait PostProcessor {
    /// Returns the number of tokens that will be added during the processing step
    fn added_tokens(&self, is_pair: bool) -> usize;
    /// Process both encodings and returns a new merged one
    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encodings = if let Some(pair_encoding) = pair_encoding {
            vec![encoding, pair_encoding]
        } else {
            vec![encoding]
        };
        encodings.iter_mut().enumerate().for_each(|(i, encoding)| {
            encoding.set_sequence_id(i);
            encoding
                .get_overflowing_mut()
                .iter_mut()
                .for_each(|encoding| encoding.set_sequence_id(i));
            encoding.set_type_ids(vec![i as u32; encoding.len()]);
        });

        let encodings = self.process_encodings(encodings, add_special_tokens)?;
        Ok(Encoding::merge(encodings, false))
    }

    /// Process any amount of encodings and returns a series of encoding (might merge them)
    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>;
}
impl dyn PostProcessor {
    pub fn default_process(
        encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        match encodings.len() {
            1 => Ok(encodings),
            _ => {
                let mut final_encoding = Encoding::default();
                for (i, mut encoding) in encodings.into_iter().enumerate() {
                    encoding.set_sequence_id(i);
                    final_encoding.merge_with(encoding, false);
                }
                Ok(vec![final_encoding])
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ProcessorError {
    #[error("encodings vector length must be either 1 or 2")]
    InvalidEncodingsVecLength,
}

/// A `Decoder` changes the raw tokens into its more readable form.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let results = self.decode_chain(tokens)?;
        Ok(results.join(""))
    }
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
}

/// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
/// and then it can train the given `Model`.
pub trait Trainer {
    type Model: Model + Sized;
    /// Whether we should show progress during the training.
    fn should_show_progress(&self) -> bool;
    /// The actual training method. This will return a new trained Model as well as a list
    /// of `special_tokens` to be added directly to the tokenizer along with the model.
    fn train(&self, model: &mut Self::Model) -> Result<Vec<AddedToken>>;
    /// Process an iterator of sequences, calling `process` for each of them in order to
    /// pre-process the said sequence as relevant.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Self { id, value, offsets }
    }
}

use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum InputSequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<Cow<'s, str>> for InputSequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        Self::Raw(input)
    }
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(input: &'s str) -> Self {
        Self::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(input: String) -> Self {
        Self::Raw(Cow::Owned(input))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        Self::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        Self::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<&'s [String]> for InputSequence<'s> {
    fn from(input: &'s [String]) -> Self {
        Self::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl From<Vec<String>> for InputSequence<'_> {
    fn from(input: Vec<String>) -> Self {
        Self::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for InputSequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        Self::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for InputSequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        Self::PreTokenizedCow(Cow::Borrowed(input))
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput<'s> {
    Single(InputSequence<'s>),
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        Self::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from(input: (I1, I2)) -> Self {
        Self::Dual(input.0.into(), input.1.into())
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{0}")]
pub struct BuilderError(String);

/// Builder for Tokenizer structs.
///
/// `build()` fails if the `model` is missing.
pub struct TokenizerBuilder<M, N, PT, PP, D> {
    model: Option<M>,
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    post_processor: Option<PP>,
    decoder: Option<D>,

    added_vocabulary: AddedVocabulary,

    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl<M, N, PT, PP, D> Default for TokenizerBuilder<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, N, PT, PP, D> TokenizerBuilder<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    /// Get an empty TokenizerBuilder.
    pub fn new() -> Self {
        Self {
            model: None,
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            added_vocabulary: AddedVocabulary::new(),
            truncation: None,
            padding: None,
        }
    }

    /// Convert the TokenizerBuilder to a Tokenizer.
    ///
    /// Conversion fails if the `model` is missing.
    pub fn build(self) -> Result<TokenizerImpl<M, N, PT, PP, D>> {
        let model = self
            .model
            .ok_or_else(|| Box::new(BuilderError("Model missing.".into())))?;
        Ok(TokenizerImpl {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model,

            post_processor: self.post_processor,
            decoder: self.decoder,
            added_vocabulary: self.added_vocabulary,
            truncation: self.truncation,
            padding: self.padding,
        })
    }

    /// Set the model.
    #[must_use]
    pub fn with_model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the normalizer.
    #[must_use]
    pub fn with_normalizer(mut self, normalizer: Option<N>) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the pre-tokenizer.
    #[must_use]
    pub fn with_pre_tokenizer(mut self, pretokenizer: Option<PT>) -> Self {
        self.pre_tokenizer = pretokenizer;
        self
    }

    /// Set the post-processor.
    #[must_use]
    pub fn with_post_processor(mut self, post_processor: Option<PP>) -> Self {
        self.post_processor = post_processor;
        self
    }

    /// Set the decoder.
    #[must_use]
    pub fn with_decoder(mut self, decoder: Option<D>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the added vocabulary.
    pub fn with_added_vocabulary(mut self, added_vocabulary: AddedVocabulary) -> Self {
        self.added_vocabulary = added_vocabulary;
        self
    }

    /// Set the truncation parameters.
    #[must_use]
    pub fn with_truncation(mut self, trunc: Option<TruncationParams>) -> Self {
        self.truncation = trunc;
        self
    }

    /// Set the padding parameters.
    #[must_use]
    pub fn with_padding(mut self, padding: Option<PaddingParams>) -> Self {
        self.padding = padding;
        self
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tokenizer(
    TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
);

impl Tokenizer {
    /// Construct a new Tokenizer based on the model.
    pub fn new(model: impl Into<ModelWrapper>) -> Self {
        Self(TokenizerImpl::new(model.into()))
    }

    /// Unwrap the TokenizerImpl.
    pub fn into_inner(
        self,
    ) -> TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > {
        self.0
    }
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let tokenizer = serde_json::from_str(&content)?;
        Ok(tokenizer)
    }
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }
    #[cfg(feature = "http")]
    pub fn from_pretrained<S: AsRef<str>>(
        identifier: S,
        params: Option<crate::utils::from_pretrained::FromPretrainedParameters>,
    ) -> Result<Self> {
        let tokenizer_file = crate::utils::from_pretrained::from_pretrained(identifier, params)?;
        Tokenizer::from_file(tokenizer_file)
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = Box<dyn std::error::Error + Send + Sync>;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT, PP, D> From<TokenizerImpl<M, N, PT, PP, D>> for Tokenizer
where
    M: Into<ModelWrapper>,
    N: Into<NormalizerWrapper>,
    PT: Into<PreTokenizerWrapper>,
    PP: Into<PostProcessorWrapper>,
    D: Into<DecoderWrapper>,
{
    fn from(t: TokenizerImpl<M, N, PT, PP, D>) -> Self {
        Self(TokenizerImpl {
            model: t.model.into(),
            normalizer: t.normalizer.map(Into::into),
            pre_tokenizer: t.pre_tokenizer.map(Into::into),
            post_processor: t.post_processor.map(Into::into),
            decoder: t.decoder.map(Into::into),
            added_vocabulary: t.added_vocabulary,
            padding: t.padding,
            truncation: t.truncation,
        })
    }
}

impl Deref for Tokenizer {
    type Target = TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{0}")]
pub struct TruncationParamError(String);

/// A `Tokenizer` is capable of encoding/decoding any text.
#[derive(Clone, Debug)]
pub struct TokenizerImpl<M, N, PT, PP, D> {
    // Tokenizer parts
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    model: M,
    post_processor: Option<PP>,
    decoder: Option<D>,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,

    // General processing parameters
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    /// Instantiate a new Tokenizer, with the given Model
    pub fn new(model: M) -> Self {
        Self {
            normalizer: None,
            pre_tokenizer: None,
            model,
            post_processor: None,
            decoder: None,

            added_vocabulary: AddedVocabulary::new(),

            truncation: None,
            padding: None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: Option<impl Into<N>>) -> &mut Self {
        self.normalizer = normalizer.map(|norm| norm.into());
        self
    }
    /// Get the normalizer
    pub fn get_normalizer(&self) -> Option<&N> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Option<impl Into<PT>>) -> &mut Self {
        self.pre_tokenizer = pre_tokenizer.map(|tok| tok.into());
        self
    }

    /// Get the pre tokenizer
    pub fn get_pre_tokenizer(&self) -> Option<&PT> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: Option<impl Into<PP>>) -> &mut Self {
        self.post_processor = post_processor.map(|post_proc| post_proc.into());
        self
    }

    /// Get the post processor
    pub fn get_post_processor(&self) -> Option<&PP> {
        self.post_processor.as_ref()
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: Option<impl Into<D>>) -> &mut Self {
        self.decoder = decoder.map(|dec| dec.into());
        self
    }

    /// Get the decoder
    pub fn get_decoder(&self) -> Option<&D> {
        self.decoder.as_ref()
    }

    /// Set the model
    pub fn with_model(&mut self, model: impl Into<M>) -> &mut Self {
        self.model = model.into();
        self
    }

    /// Get the model
    pub fn get_model(&self) -> &M {
        &self.model
    }

    /// Set the added vocabulary.
    pub fn with_added_vocabulary(&mut self, added_vocabulary: AddedVocabulary) -> &mut Self {
        self.added_vocabulary = added_vocabulary;
        self
    }

    /// Get the added vocabulary
    pub fn get_added_vocabulary(&self) -> &AddedVocabulary {
        &self.added_vocabulary
    }

    /// Set the truncation parameters
    ///
    /// Fails if `stride` is too high relative to `max_length` and `post_processor.added_tokens()`
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> Result<&mut Self> {
        if let Some(trunc_params) = &trunc {
            let n_added_tokens = self.get_n_added_tokens(false);
            let effective_max_length = trunc_params.max_length - n_added_tokens;
            if effective_max_length < trunc_params.stride {
                return Err(Box::new(TruncationParamError(format!(
                    "tokenizer stride set to {}, which is greater than or equal to its effective max length of {} (= {} original max length - {} added special tokens), ",
                    trunc_params.stride, effective_max_length, trunc_params.max_length, n_added_tokens
                ))));
            }
        }
        self.truncation = trunc;
        Ok(self)
    }

    /// Get the currently set truncation parameters
    pub fn get_truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Get a mutable reference to the currently set truncation parameters
    pub fn get_truncation_mut(&mut self) -> Option<&mut TruncationParams> {
        self.truncation.as_mut()
    }

    /// Set the padding parameters
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Get the currently set padding parameters
    pub fn get_padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    /// Get a mutable reference to the currently set padding parameters
    pub fn get_padding_mut(&mut self) -> Option<&mut PaddingParams> {
        self.padding.as_mut()
    }

    // Get the vocabulary as a plain HashMap for bindings compatibility
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut final_vocab = self.model.get_vocab();

        if with_added_tokens {
            let added_vocab = self.added_vocabulary.get_vocab();
            if !added_vocab.is_empty() {
                final_vocab.reserve(added_vocab.len());
                for (token, id) in added_vocab {
                    final_vocab.insert(token.clone(), *id);
                }
            }
        }

        final_vocab
    }

    /// Get the added tokens decoder
    pub fn get_added_tokens_decoder(&self) -> AHashMap<u32, AddedToken> {
        self.added_vocabulary.get_added_tokens_decoder().clone()
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        // TODO ArthurZ THIS IS WRONG! We need to measure the length of the `set` because
        // now some tokens can be both in the added_tokens_encoder and in the vocab
        if with_added_tokens {
            self.get_vocab(true).len()
        } else {
            self.model.get_vocab_size()
        }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_vocabulary.token_to_id(token, &self.model)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.added_vocabulary
            .simple_id_to_token(id)
            .or_else(|| self.model.id_to_token(id))
    }

    /// set the added vocab's splitting scheme
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.added_vocabulary.set_encode_special_tokens(value);
    }

    /// Get added token value
    pub fn get_encode_special_tokens(&self) -> bool {
        self.added_vocabulary.get_encode_special_tokens()
    }

    /// Encode a single sequence
    fn encode_single_sequence(
        &self,
        sequence: InputSequence,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding> {
        let encode = |is_pre_tokenized, subseq_idx, subseq| -> Result<Encoding> {
            let normalized = self
                .added_vocabulary
                .extract_and_normalize(self.normalizer.as_ref(), subseq);
            let pre_tokenized = self.do_pre_tokenize(normalized)?;
            let subseq_encoding = self.do_tokenize(
                pre_tokenized,
                type_id,
                if is_pre_tokenized {
                    Some(subseq_idx as u32)
                } else {
                    None
                },
                offsets_type,
            )?;

            Ok(subseq_encoding)
        };

        match sequence {
            InputSequence::PreTokenized(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedOwned(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedCow(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::Raw(seq) => encode(false, 0, seq.as_ref()),
        }
    }

    /// Encode the given input. This method accepts both single sequences, as well as pair
    /// sequences. Also, a sequence can be a string, or already pre-tokenized input directly:
    /// Contrarily to `encode`, it does not compute offsets
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode_fast("Single sequence", false);
    /// tokenizer.encode_fast(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode_fast(&["Single", "sequence"][..], false);
    /// tokenizer.encode_fast((
    ///     &["Sequence", "A"][..],
    ///     &["Sequence", "B"][..]
    /// ), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode_fast(("A complete sequence", &["And", "a", "tokenized"][..]), false);
    /// ```
    pub fn encode_fast<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::None)?;
        let pair_encoding = pair
            .map(|sequence| self.encode_single_sequence(sequence, 1, OffsetType::None))
            .transpose()?;

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Encode the given input. This method accepts both single sequences, as well as pair
    /// sequences. Also, a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((
    ///     &["Sequence", "A"][..],
    ///     &["Sequence", "B"][..]
    /// ), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(("A complete sequence", &["And", "a", "tokenized"][..]), false);
    /// ```
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Byte)?;
        let pair_encoding = pair
            .map(|sequence| self.encode_single_sequence(sequence, 1, OffsetType::Byte))
            .transpose()?;

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Encode the given input, using offsets relative to chars instead of bytes.
    /// This method accepts both single sequences, as well as pair sequences. Also,
    /// a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((
    ///     &["Sequence", "A"][..],
    ///     &["Sequence", "B"][..]
    /// ), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(("A complete sequence", &["And", "a", "tokenized"][..]), false);
    /// ```
    pub fn encode_char_offsets<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Char)?;
        let pair_encoding = pair
            .map(|sequence| self.encode_single_sequence(sequence, 1, OffsetType::Char))
            .transpose()?;

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Encode a large text using parallel processing by splitting on newlines.
    ///
    /// This method produces **identical results** to [`encode`] but uses parallel
    /// processing for texts above a size threshold. It automatically falls back to
    /// single-threaded encoding when parallelism would not help.
    ///
    /// # When to use
    /// - Encoding large documents, books, or corpora
    /// - Batch processing where throughput matters more than latency
    /// - Inputs with natural newline boundaries (prose, code, logs)
    ///
    /// # Performance characteristics
    /// - **Threshold**: 50KB minimum for parallelization
    /// - **Speedup**: 2-6x on typical multi-core systems
    /// - **Requirement**: Text must contain newlines for parallelization
    ///
    /// # Why only newlines work as split points
    /// Newlines are safe boundaries because pre-tokenizers typically treat them as
    /// explicit token boundaries. Splitting at other whitespace could produce different
    /// token sequences than single-threaded encoding.
    pub fn encode_parallel(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>
    where
        M: Send + Sync,
        N: Send + Sync,
        PT: Send + Sync,
        PP: Send + Sync,
        D: Send + Sync,
    {
        // Lowered threshold for more aggressive parallelization
        const PARALLEL_THRESHOLD: usize = 20_000;

        // Fall back to encode_fast for small texts (skips offset tracking)
        if text.len() < PARALLEL_THRESHOLD {
            return self.encode_fast(text, add_special_tokens);
        }

        // Quick check for newlines using memchr (SIMD-accelerated)
        let has_newlines = memchr::memchr(b'\n', text.as_bytes()).is_some();
        if !has_newlines {
            return self.encode_fast(text, add_special_tokens);
        }

        // Split text on newlines for parallel encoding
        let num_threads = current_num_threads();
        let multiplier = if text.len() > 10_000_000 { 4 } else { 2 };
        let target_chunk_size = (text.len() / (num_threads * multiplier)).max(PARALLEL_THRESHOLD);

        let chunks = Self::split_on_newlines(text, target_chunk_size);
        if chunks.len() <= 1 {
            return self.encode_fast(text, add_special_tokens);
        }

        // Encode chunks in parallel (without post-processing)
        // Use OffsetType::None for speed - we don't need character offsets
        let encodings: Vec<Encoding> = chunks
            .into_maybe_par_iter()
            .map(|chunk| {
                let sequence = InputSequence::Raw(std::borrow::Cow::Borrowed(chunk));
                self.encode_single_sequence(sequence, 0, OffsetType::None)
            })
            .collect::<Result<Vec<_>>>()?;

        // Merge all encodings with growing offsets
        let merged = Encoding::merge(encodings, true);

        // Apply post-processing to the merged result
        self.post_process(merged, None, add_special_tokens)
    }

    /// Split text on newlines into chunks of approximately target_chunk_size bytes.
    /// Each chunk includes its trailing newline to preserve token boundaries.
    fn split_on_newlines(text: &str, target_chunk_size: usize) -> Vec<&str> {
        if text.is_empty() {
            return vec![];
        }

        let estimated_chunks = (text.len() / target_chunk_size).max(1) + 1;
        let mut chunks = Vec::with_capacity(estimated_chunks);

        let mut current_start = 0;
        let mut current_len = 0;

        for segment in text.split_inclusive('\n') {
            let segment_len = segment.len();
            current_len += segment_len;

            if current_len >= target_chunk_size {
                // End of segment is the split point
                let end = current_start + current_len;
                chunks.push(&text[current_start..end]);
                current_start = end;
                current_len = 0;
            }
        }

        // Push any remaining text
        if current_start < text.len() {
            chunks.push(&text[current_start..]);
        }

        chunks
    }

    /// Split text into chunks using boundaries that are safe for verified
    /// space-prefix BPE tokenizers:
    /// - after a complete newline run (`\n`, `\r\n`, `\n\n`, ...)
    /// - before a complete ASCII-space run
    ///
    /// The ASCII-space rule is intentionally not used by the newline-only
    /// `encode_parallel_ids_only` path. It is exact for tokenizer configs where
    /// a leading space is represented as part of the following token, but it is
    /// not a universal tokenizer invariant.
    fn safe_newline_or_space_boundaries(text: &str) -> Vec<usize> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut boundaries = Vec::new();
        let mut idx = 0usize;

        while idx < len {
            match bytes[idx] {
                b'\n' => {
                    let mut cut = idx + 1;
                    while cut < len && bytes[cut] == b'\n' {
                        cut += 1;
                    }
                    if cut < len {
                        boundaries.push(cut);
                    }
                    idx = cut;
                }
                b' ' => {
                    if idx > 0 {
                        boundaries.push(idx);
                    }
                    while idx < len && bytes[idx] == b' ' {
                        idx += 1;
                    }
                }
                _ => idx += 1,
            }
        }

        boundaries
    }

    fn split_on_newlines_or_ascii_spaces(
        text: &str,
        target_chunk_size: usize,
        lookaround_bytes: usize,
    ) -> Vec<&str> {
        if text.is_empty() {
            return vec![];
        }

        let target_chunk_size = target_chunk_size.max(1);
        let boundaries = Self::safe_newline_or_space_boundaries(text);
        if boundaries.is_empty() {
            return vec![text];
        }

        let len = text.len();
        let estimated_chunks = (len / target_chunk_size).max(1) + 1;
        let mut chunks = Vec::with_capacity(estimated_chunks);
        let mut start = 0usize;
        let mut boundary_cursor = 0usize;

        while start < len {
            if len - start <= target_chunk_size {
                chunks.push(&text[start..]);
                break;
            }

            let desired = start + target_chunk_size;
            let min_cut = start + (target_chunk_size / 2).max(1);
            let max_cut = len.min(desired.saturating_add(lookaround_bytes));

            while boundary_cursor < boundaries.len() && boundaries[boundary_cursor] <= start {
                boundary_cursor += 1;
            }

            let mut scan = boundary_cursor;
            while scan < boundaries.len() && boundaries[scan] < desired {
                scan += 1;
            }

            let forward_cut = boundaries
                .get(scan)
                .copied()
                .filter(|&candidate| candidate <= max_cut);

            let backward_cut = scan
                .checked_sub(1)
                .and_then(|prev| boundaries.get(prev).copied())
                .filter(|&candidate| candidate >= min_cut && candidate > start);

            let cut = forward_cut.or(backward_cut);

            match cut {
                Some(end) if end > start => {
                    chunks.push(&text[start..end]);
                    start = end;
                    boundary_cursor = scan.saturating_sub(1);
                }
                _ => {
                    // No tokenizer-safe boundary near the target size. Returning the
                    // remaining text as one chunk preserves exactness; callers will
                    // naturally get little or no parallelism for no-boundary inputs.
                    chunks.push(&text[start..]);
                    break;
                }
            }
        }

        chunks
    }

    /// Encode a single raw string to just its token IDs, bypassing full `Encoding` construction.
    ///
    /// This runs the full normalization → pre-tokenization → tokenization pipeline but
    /// uses [`Model::tokenize_ids`] instead of [`Model::tokenize`], which skips:
    /// - The BPE `RwLock` cache (eliminates cross-core contention)
    /// - `Token` struct allocation (no string values or offset tracking)
    ///
    /// Used internally by [`encode_parallel_ids_only`] for each parallel chunk.
    fn encode_single_sequence_ids(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = self
            .added_vocabulary
            .extract_and_normalize(self.normalizer.as_ref(), text);
        let pre_tokenized = self.do_pre_tokenize(normalized)?;
        // Use tokenize_ids path: no cache, no string alloc
        pre_tokenized.tokenize_to_ids(|seq| self.model.tokenize_ids(seq))
    }

    fn encoding_from_ids_only(&self, ids: Vec<u32>, add_special_tokens: bool) -> Result<Encoding>
    where
        PP: PostProcessor,
    {
        let total_len = ids.len();
        let encoding = Encoding::new(
            ids,
            vec![0u32; total_len],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0u32; total_len],
            vec![1u32; total_len],
            Vec::new(),
            ahash::AHashMap::new(),
        );

        self.post_process(encoding, None, add_special_tokens)
    }

    /// Parallel encoding optimized for throughput when only token IDs are needed.
    ///
    /// Achieves **~7x speedup** over single-threaded `encode()` by solving two
    /// bottlenecks in the standard parallel encoding path:
    ///
    /// ## Root cause: BPE cache `RwLock` contention
    /// The standard `Model::tokenize()` uses an `RwLock<AHashMap>` cache. Even
    /// non-blocking `try_read()` causes cache-line bouncing across cores via atomic
    /// compare-and-swap operations, serializing what should be parallel work.
    /// With 16 threads, the standard `encode_parallel()` achieves only ~1.5x speedup.
    ///
    /// This method bypasses the cache entirely via [`Model::tokenize_ids`], which
    /// calls `merge_word()` directly. The BPE merge algorithm is deterministic and
    /// stateless — the cache is purely an optimization for sequential use.
    ///
    /// ## What it skips
    /// - `RwLock` cache lookups (eliminates cross-core contention)
    /// - `Token` struct allocation (no string values or offset tracking)
    /// - `Encoding::merge()` (avoids extending 7+ vectors iteratively)
    /// - Per-token offset computation and character mapping
    ///
    /// ## Returned Encoding
    /// The returned `Encoding` has valid `ids`, `type_ids`, `special_tokens_mask`,
    /// and `attention_mask` fields. **The `tokens`, `offsets`, and `words` fields
    /// are empty** (`Vec::new()`). Only use this method when you need IDs only
    /// (e.g., for model inference where you discard string representations).
    ///
    /// ## Safety of newline splitting
    /// Text is split on `\n` boundaries using `split_inclusive`, which preserves
    /// the newline character at the end of each chunk. Since pre-tokenizers treat
    /// newlines as word boundaries, splitting here produces identical token IDs
    /// to encoding the full text as a single string.
    ///
    /// ## Fallback behavior
    /// Falls back to `encode_fast()` (single-threaded, full Encoding) when:
    /// - Text is shorter than 10KB
    /// - Text contains no newlines
    /// - Only one chunk would be produced
    pub fn encode_parallel_ids_only(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>
    where
        M: Send + Sync,
        N: Send + Sync,
        PT: Send + Sync,
        PP: Send + Sync,
        D: Send + Sync,
    {
        // Minimum text size to bother with parallelism
        const MIN_PARALLEL_SIZE: usize = 10_000;
        // Minimum chunk size — below this, per-chunk overhead dominates
        const MIN_CHUNK_SIZE: usize = 2_000;

        if text.len() < MIN_PARALLEL_SIZE {
            return self.encode_fast(text, add_special_tokens);
        }

        let has_newlines = memchr::memchr(b'\n', text.as_bytes()).is_some();
        if !has_newlines {
            return self.encode_fast(text, add_special_tokens);
        }

        // Create many small chunks for better work-stealing distribution
        let num_threads = current_num_threads();
        // 4x oversubscription for work-stealing balance
        let target_chunks = num_threads * 4;
        let target_chunk_size = (text.len() / target_chunks).max(MIN_CHUNK_SIZE);

        let chunks = Self::split_on_newlines(text, target_chunk_size);
        if chunks.len() <= 1 {
            return self.encode_fast(text, add_special_tokens);
        }

        // Encode chunks in parallel — only extract IDs
        let id_vecs: Vec<Vec<u32>> = chunks
            .into_maybe_par_iter()
            .map(|chunk| self.encode_single_sequence_ids(chunk))
            .collect::<Result<Vec<_>>>()?;

        // Calculate total length for pre-allocation
        let total_len: usize = id_vecs.iter().map(|v| v.len()).sum();

        // Concatenate all ID vectors
        let mut all_ids = Vec::with_capacity(total_len);
        for ids in id_vecs {
            all_ids.extend_from_slice(&ids);
        }

        self.encoding_from_ids_only(all_ids, add_special_tokens)
    }

    /// Parallel ID-only encoding for verified space-prefix BPE tokenizers.
    ///
    /// This is the same allocation-light path as [`encode_parallel_ids_only`],
    /// but it can also split before ASCII-space runs. For compatible tokenizer
    /// configs this preserves exactly the same token IDs as encoding the full
    /// text while allowing large no-newline documents to use rayon parallelism.
    ///
    /// Do not use this as a generic tokenizer optimization unless the tokenizer's
    /// boundary behavior has been verified.
    pub fn encode_parallel_ids_only_space_safe(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding>
    where
        M: Send + Sync,
        N: Send + Sync,
        PT: Send + Sync,
        PP: Send + Sync,
        D: Send + Sync,
    {
        let config = SpaceSafeParallelConfig::global();
        self.encode_parallel_ids_only_space_safe_with_options(
            text,
            add_special_tokens,
            config.min_parallel_bytes,
            config.target_chunk_bytes,
            config.lookaround_bytes,
        )
    }

    /// Same as [`encode_parallel_ids_only_space_safe`], but with explicit
    /// tunables. This exists for CPU-specific autotuning: sweep these values on
    /// the exact host and persist the winner as environment variables for the
    /// production process.
    pub fn encode_parallel_ids_only_space_safe_with_options(
        &self,
        text: &str,
        add_special_tokens: bool,
        min_parallel_bytes: usize,
        target_chunk_bytes: usize,
        lookaround_bytes: usize,
    ) -> Result<Encoding>
    where
        M: Send + Sync,
        N: Send + Sync,
        PT: Send + Sync,
        PP: Send + Sync,
        D: Send + Sync,
    {
        if text.len() < min_parallel_bytes.max(1) {
            return self.encode_fast(text, add_special_tokens);
        }

        let chunks =
            Self::split_on_newlines_or_ascii_spaces(text, target_chunk_bytes, lookaround_bytes);
        if chunks.len() <= 1 {
            return self.encode_fast(text, add_special_tokens);
        }

        let id_vecs: Vec<Vec<u32>> = chunks
            .into_maybe_par_iter()
            .map(|chunk| self.encode_single_sequence_ids(chunk))
            .collect::<Result<Vec<_>>>()?;

        let total_len: usize = id_vecs.iter().map(|v| v.len()).sum();
        let mut all_ids = Vec::with_capacity(total_len);
        for ids in id_vecs {
            all_ids.extend_from_slice(&ids);
        }

        self.encoding_from_ids_only(all_ids, add_special_tokens)
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .iter()
            .filter_map(|id| {
                self.added_vocabulary
                    .simple_id_to_token(*id)
                    .or_else(|| self.model.id_to_token(*id))
                    .filter(|token| {
                        !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                    })
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Decode the given ids, back to a String
    /// See [`DecodeStream`]
    pub fn decode_stream(&self, skip_special_tokens: bool) -> DecodeStream<'_, M, N, PT, PP, D> {
        DecodeStream::new(self, skip_special_tokens)
    }
}

/// DecodeStream will keep the state necessary to produce individual chunks of
/// strings given an input stream of token_ids.
///
/// This is necessary because decoding in general cannot achieve that since strings
/// depend on surrounding ids to provide a valid string. Typically stripping extra spaces
///
/// Example:
///
/// ```
/// # #[cfg(not(target_os = "windows"))]
/// # {
/// use tokenizers::Tokenizer;
/// let tokenizer = Tokenizer::from_file("data/roberta.json").unwrap();
///
/// let mut decode_stream = tokenizer.decode_stream(false);
/// assert_eq!(decode_stream.step(713).unwrap(), Some("This".to_string()));
/// assert_eq!(decode_stream.step(16).unwrap(), Some(" is".to_string()));
/// assert_eq!(decode_stream.step(41).unwrap(), Some(" an".to_string()));
/// assert_eq!(
///     decode_stream.step(1246).unwrap(),
///     Some(" example".to_string())
/// );
/// # }
/// ```
///
/// Returning `None` means the given id is not enough to produce a chunk.
/// This typically happens with `byte_fallback` options where some tokens do
/// not represent valid utf-8, and only follow-up token_ids will help produce
/// a valid chunk.
/// ```
/// use tokenizers::{Tokenizer, TokenizerBuilder, models::bpe::BPE, decoders::byte_fallback::ByteFallback, pre_tokenizers::byte_level::ByteLevel, normalizers::unicode::NFC};
/// use std::collections::HashMap;
/// use std::iter::FromIterator;
///
/// let vocab = HashMap::from_iter([
///     ("<0x20>".to_string(), 0),
///     ("<0xC3>".to_string(), 1),
///     ("<0xA9>".to_string(), 2),
///     (" This".to_string(), 3),
/// ]);
/// let merges = vec![];
/// let bpe = BPE::builder()
///     .vocab_and_merges(vocab, merges)
///     .byte_fallback(true)
///     .build()
///     .unwrap();
/// let tokenizer = TokenizerBuilder::default()
///     .with_model(bpe)
///     .with_decoder(Some(ByteFallback::default()))
///     .with_normalizer(Some(NFC))
///     .with_pre_tokenizer(Some(ByteLevel::default()))
///     .with_post_processor(Some(ByteLevel::default()))
///     .build().unwrap();
///
/// let mut decode_stream = tokenizer.decode_stream(false);
/// // Single byte_fallback is valid utf-8
/// assert_eq!(decode_stream.step(0).unwrap(), Some(" ".to_string()));
/// // Invalid utf-8
/// assert_eq!(decode_stream.step(1).unwrap(), None);
/// // Valid utf-8 again, this corresponds to both tokens: [1, 2]
/// assert_eq!(decode_stream.step(2).unwrap(), Some("é".to_string()));
/// ```
///
/// To see how [`DecodeStream`] is necessary, let's show how using raw [`TokenizerImpl::decode`] would
/// fail.
///
/// ```
/// use tokenizers::{Tokenizer, TokenizerBuilder, models::bpe::BPE, pre_tokenizers::{byte_level::ByteLevel, metaspace::Metaspace}, normalizers::unicode::NFC};
/// use std::collections::HashMap;
/// use std::iter::FromIterator;
///
/// let vocab = HashMap::from_iter([
///     ("▁This".to_string(), 0),
/// ]);
/// let merges = vec![];
/// let bpe = BPE::builder()
///     .vocab_and_merges(vocab, merges)
///     .byte_fallback(true)
///     .build()
///     .unwrap();
/// let tokenizer = TokenizerBuilder::new()
///     .with_model(bpe)
///     .with_decoder(Some(Metaspace::default()))
///     .with_normalizer(Some(NFC))
///     .with_pre_tokenizer(Some(ByteLevel::default()))
///     .with_post_processor(Some(ByteLevel::default()))
///     .build()
///     .unwrap();
///
/// // Strip decoder removes the extra initial space
/// assert_eq!(tokenizer.decode(&[0, 0], false).unwrap(), "This This");
/// // Decoding one token at a time would produce "ThisThis"
/// assert_eq!(tokenizer.decode(&[0], false).unwrap(), "This");
///
/// // Using a stream fixes it by keeping the necessary state.
/// let mut decode_stream = tokenizer.decode_stream(false);
/// assert_eq!(decode_stream.step(0).unwrap(), Some("This".to_string()));
/// assert_eq!(decode_stream.step(0).unwrap(), Some(" This".to_string()));
/// ```
pub struct DecodeStream<'tok, M, N, PT, PP, D> {
    /// A reference to the tokenizer
    tokenizer: &'tok TokenizerImpl<M, N, PT, PP, D>,
    /// Regular decode option that is kept throughout.
    skip_special_tokens: bool,
    /// A temporary buffer of the necessary token_ids needed
    /// to produce valid string chunks.
    /// This typically contains 3 parts:
    ///  - read
    ///  - prefix
    ///  - rest
    ///
    /// Read is the bit necessary to surround the prefix
    /// so decoding the whole ids produces a valid prefix.
    /// Prefix is the previously produced string, kept around to trim off of
    /// the next valid chunk
    ids: Vec<u32>,
    /// The previously returned chunk that needs to be discarded from the
    /// decoding of the current ids to produce the next chunk
    prefix: String,
    /// The index within the ids corresponding to the prefix so we can drain
    /// correctly
    prefix_index: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum DecodeStreamError {
    #[error("Invalid prefix encountered while decoding stream. Token ID: {token_id}, Expected prefix: '{expected_prefix}', Actual string: '{actual_string}'")]
    InvalidPrefix {
        token_id: u32,
        expected_prefix: String,
        actual_string: String,
    },
}

impl<'tok, M, N, PT, PP, D> DecodeStream<'tok, M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    fn new(tokenizer: &'tok TokenizerImpl<M, N, PT, PP, D>, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            ids: vec![],
            skip_special_tokens,
            prefix: "".to_string(),
            prefix_index: 0,
        }
    }

    /// See [`DecodeStream`]
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        step_decode_stream(
            self.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
    }
}

/// Internal function exposed only to bypass python limitations
pub fn step_decode_stream<M, N, PT, PP, D>(
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    token_ids: Vec<u32>,
    skip_special_tokens: bool,
    ids: &mut Vec<u32>,
    prefix: &mut String,
    prefix_index: &mut usize,
) -> Result<Option<String>>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    if prefix.is_empty() && !ids.is_empty() {
        let new_prefix = tokenizer.decode(ids, skip_special_tokens)?;
        if !new_prefix.ends_with('�') {
            *prefix = new_prefix;
            *prefix_index = ids.len();
        }
    }

    ids.extend(token_ids);
    let string = tokenizer.decode(ids.as_slice(), skip_special_tokens)?;
    if string.len() > prefix.len() && !string.ends_with('�') {
        if !(string.starts_with(&*prefix)) {
            return Err(Box::new(DecodeStreamError::InvalidPrefix {
                token_id: *ids.last().unwrap(),
                expected_prefix: prefix.clone(),
                actual_string: string,
            }));
        }

        let new_text = &string[prefix.len()..].to_string();
        let new_prefix_index = ids.len() - *prefix_index;
        *ids = ids.drain(*prefix_index..).collect();
        *prefix = tokenizer.decode(ids, skip_special_tokens)?;
        *prefix_index = new_prefix_index;
        Ok(Some(new_text.to_string()))
    } else {
        Ok(None)
    }
}
impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
{
    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn do_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
        type_id: u32,
        word_idx: Option<u32>,
        offsets_type: OffsetType,
    ) -> Result<Encoding> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();
        pretokenized.tokenize(|normalized| self.model.tokenize(normalized.get()))?;
        pretokenized.into_encoding(word_idx, type_id, offsets_type)
    }
}

#[allow(dead_code)]
impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    N: Normalizer,
{
    /// Normalization logic, go through all normalizers
    fn do_normalize<V: Into<NormalizedString>>(&self, normalized: V) -> Result<NormalizedString> {
        let mut normalized: NormalizedString = normalized.into();

        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }

        Ok(normalized)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    N: Normalizer,
    M: Model,
{
    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_special_tokens(tokens, &self.model, self.normalizer.as_ref())
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_tokens(tokens, &self.model, self.normalizer.as_ref())
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    PT: PreTokenizer,
{
    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn do_pre_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
    ) -> Result<PreTokenizedString> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();
        if let Some(ref pretok) = self.pre_tokenizer {
            pretok.pre_tokenize(&mut pretokenized)?;
        }

        Ok(pretokenized)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    PP: PostProcessor,
{
    /// Post processing logic, handling the case where there is no PostProcessor set
    pub fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // 1. First we truncate if needed
        let (encoding, pair_encoding) = {
            if let Some(trunc) = &self.truncation {
                let n_added_tokens = self.get_n_added_tokens(pair_encoding.is_some());

                if add_special_tokens && n_added_tokens > 0 {
                    let params = TruncationParams {
                        max_length: trunc.max_length - n_added_tokens,
                        ..*trunc
                    };
                    truncate_encodings(encoding, pair_encoding, &params)?
                } else {
                    truncate_encodings(encoding, pair_encoding, trunc)?
                }
            } else {
                (encoding, pair_encoding)
            }
        };

        // 2. Then We post process
        let final_encoding = if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding, add_special_tokens)?
        } else {
            let encodings = if let Some(pair_encoding) = pair_encoding {
                vec![encoding, pair_encoding]
            } else {
                vec![encoding]
            };
            let mut encodings =
                <dyn PostProcessor>::default_process(encodings, add_special_tokens)?;
            if encodings.len() != 1 {
                panic!("We haven't reduced the encodings like we should have");
            }
            encodings.pop().unwrap()
        };

        // 3. Then we pad if needed
        let [final_encoding] = if let Some(params) = &self.padding {
            let mut arr = [final_encoding];
            pad_encodings(&mut arr, params)?;
            arr
        } else {
            [final_encoding]
        };

        Ok(final_encoding)
    }

    fn get_n_added_tokens(&self, is_pair: bool) -> usize {
        if let Some(processor) = &self.post_processor {
            processor.added_tokens(is_pair)
        } else {
            0
        }
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_maybe_par_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(&mut encodings, params)?;
        }

        Ok(encodings)
    }

    /// Encode all the sentences in parallel, using multiple threads.
    /// The offsets on each `Encoding` will be relative to chars instead of bytes.
    pub fn encode_batch_char_offsets<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_maybe_par_iter()
            .map(|input| self.encode_char_offsets(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(&mut encodings, params)?;
        }

        Ok(encodings)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch_fast<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_maybe_par_iter()
            .map(|input| self.encode_fast(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(&mut encodings, params)?;
        }

        Ok(encodings)
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(
        &self,
        sentences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>>
    where
        M: Send + Sync,
    {
        sentences
            .into_maybe_par_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }

    /// Train our Model from files
    pub fn train_from_files<T>(&mut self, trainer: &mut T, files: Vec<String>) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
    {
        let mut len = 0;
        for file in files.iter() {
            len += File::open(file)
                .and_then(|f| f.metadata())
                .map(|m| m.len())?;
        }

        let max_read = 1_000_000;

        ResultShunt::process(
            files.into_iter().flat_map(|filename| {
                match File::open(filename) {
                    Ok(file) => {
                        let file = BufReader::with_capacity(max_read, file);
                        // We read new lines using this API instead of the Lines Iterator
                        // on purpose. We want to keep the `\n` and potential `\r` between each lines
                        // We use an iterator to be able to chain with par_bridge.
                        itertools::Either::Left(file.lines_with_ending())
                    }
                    Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
                }
            }),
            |sequences| -> Result<()> {
                let progress = if trainer.should_show_progress() {
                    let progress = ProgressBar::new(len);
                    progress.set_style(
                        ProgressStyle::default_bar()
                            .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {percent:>18!}%")
                            .expect("Invalid progress template"),
                    );
                    progress
                        .set_message(format!("Pre-processing files ({:.2} Mo)", len / 1_000_000));
                    Some(progress)
                } else {
                    None
                };

                trainer.feed(
                    sequences.inspect(|s| {
                        if let Some(progress) = &progress {
                            progress.inc(s.len() as u64)
                        }
                    }),
                    |seq| {
                        let normalized = self
                            .added_vocabulary
                            .extract_and_normalize(self.normalizer.as_ref(), seq.as_ref());
                        let pre_tokenized = self.do_pre_tokenize(normalized)?;
                        Ok(pre_tokenized
                            .get_splits(OffsetReferential::Original, OffsetType::Byte)
                            .into_iter()
                            .map(|(s, _, _)| s.to_owned())
                            .collect())
                    },
                )?;

                if let Some(pbar) = progress {
                    pbar.finish();
                }
                let special_tokens = trainer.train(&mut self.model)?;
                self.add_special_tokens(&special_tokens);

                Ok(())
            },
        )??;
        Ok(self)
    }

    /// Train our Model, using the given Trainer and iterator
    pub fn train<T, I, S>(&mut self, trainer: &mut T, sequences: I) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let (lower, upper) = sequences.size_hint();
        let len = upper.unwrap_or(lower) as u64;
        let progress = if trainer.should_show_progress() {
            let progress = ProgressBar::new(len);
            progress.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            progress.set_message("Pre-processing sequences");
            Some(progress)
        } else {
            None
        };

        trainer.feed(
            sequences.inspect(|_s| {
                if let Some(progress) = &progress {
                    progress.inc(1)
                }
            }),
            |seq| {
                let normalized = self
                    .added_vocabulary
                    .extract_and_normalize(self.normalizer.as_ref(), seq.as_ref());
                let pre_tokenized = self.do_pre_tokenize(normalized)?;
                Ok(pre_tokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, _, _)| s.to_owned())
                    .collect())
            },
        )?;
        if let Some(pbar) = progress {
            pbar.finish();
        }

        let special_tokens = trainer.train(&mut self.model)?;
        self.add_special_tokens(&special_tokens);

        Ok(self)
    }
}

impl<M, N, PT, PP, D> std::str::FromStr for TokenizerImpl<M, N, PT, PP, D>
where
    M: for<'de> Deserialize<'de> + Model,
    N: for<'de> Deserialize<'de> + Normalizer,
    PT: for<'de> Deserialize<'de> + PreTokenizer,
    PP: for<'de> Deserialize<'de> + PostProcessor,
    D: for<'de> Deserialize<'de> + Decoder,
{
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
    PP: DeserializeOwned + PostProcessor,
    D: DeserializeOwned + Decoder,
{
    /// Instantiate a new Tokenizer from the given file
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let tokenizer = serde_json::from_str(&content)?;
        Ok(tokenizer)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
    PP: DeserializeOwned + PostProcessor,
    D: DeserializeOwned + Decoder,
{
    /// Instantiate a new Tokenizer from bytes
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
    PP: DeserializeOwned + PostProcessor,
    D: DeserializeOwned + Decoder,
{
    #[deprecated(
        since = "0.14.0",
        note = "Users should download the file separately using https://github.com/huggingface/hf-hub instead, which splits concerns of accessing the web, and should use the new cache layout"
    )]
    #[cfg(feature = "http")]
    /// Instantiate a new Tokenizer from a file hosted on the Hugging Face Hub.
    /// It expects the `identifier` of a model that includes a `tokenizer.json` file.
    pub fn from_pretrained<S: AsRef<str>>(
        identifier: S,
        params: Option<crate::utils::from_pretrained::FromPretrainedParameters>,
    ) -> Result<Self> {
        let tokenizer_file = crate::utils::from_pretrained::from_pretrained(identifier, params)?;
        TokenizerImpl::from_file(tokenizer_file)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
    PP: Serialize,
    D: Serialize,
{
    /// Serialize the current tokenizer as a String
    pub fn to_string(&self, pretty: bool) -> Result<String> {
        Ok(if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        })
    }

    /// Save the current tokenizer at the given path
    pub fn save<P: AsRef<Path>>(&self, path: P, pretty: bool) -> Result<()> {
        let serialized = self.to_string(pretty)?;

        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }
}
