//! Fixed-width value codec and process-wide gate for MDBX `DUP_FIXED` storage tables.
//!
//! # Background
//!
//! reth stores each account's storage as one MDBX `DUP_SORT` sub-tree keyed by account. Under a
//! hot account with millions of randomly-keyed slots this nested sub-tree degrades badly. The
//! value of the fixed-size storage tables ([`HashedStorages`](crate::tables::HashedStorages) and
//! [`PlainStorageState`](crate::tables::PlainStorageState)) is a
//! [`StorageEntry`](reth_primitives_traits::StorageEntry): a 32-byte subkey plus a `U256` value.
//! MDBX `DUP_FIXED` stores same-size duplicates as a packed fixed array rather than a nested
//! B-tree, which can be far cheaper for these tables.
//!
//! # The correctness bar
//!
//! `DUP_FIXED` requires *every* duplicate value in a table to be exactly the same byte length.
//! reth's stock `Compact` encoding of `StorageEntry` trims the leading zeros of the `U256` value,
//! so it is variable-length and illegal under `DUP_FIXED`.
//!
//! `StorageEntry`'s compact encoding is exactly `key[32] ++ minimal_big_endian(value)` (see the
//! manual `Compact` impl in `reth-primitives-traits`; the `value` part is produced by `U256`'s
//! `Compact` impl, which strips leading zero bytes). Because `StorageEntry` lives in an external
//! crate we cannot change its codec directly, so instead we *reshape the compressed bytes* at the
//! MDBX read/write seam when the gate is on:
//!
//! - [`expand_buf`]: `key[32] ++ minimal_be(value)` -> `key[32] ++ be32(value)` (constant 64 bytes)
//! - [`contract`]: `key[32] ++ be32(value)` -> `key[32] ++ minimal_be(value)`
//!
//! The contracted form is byte-identical to what stock `Compress` would have produced, so the
//! stock `Decompress` reconstructs the exact same `StorageEntry` and the round-trip is lossless.
//!
//! # Gating (opt-in, default OFF)
//!
//! `DUP_FIXED` is a create-time table flag, so this is gated once per process. The gate is a
//! process-wide flag initialised from the `RETH_STORAGE_DUPFIXED` environment variable
//! (`RETH_STORAGE_DUPFIXED=1` turns it on) on first use, and can be overridden programmatically
//! via [`set_enabled`] (used by tests). Default is OFF, which is byte-identical to stock reth
//! (normal `DUP_SORT`, compact values).

use std::sync::atomic::{AtomicU8, Ordering};

/// Fixed on-disk width of a `StorageEntry` value under `DUP_FIXED`: 32-byte subkey + 32-byte
/// big-endian `U256` value.
pub(crate) const FIXED_WIDTH: usize = 64;

const KEY_LEN: usize = 32;

// Tri-state gate: 0 = uninitialised, 1 = off, 2 = on.
const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static GATE: AtomicU8 = AtomicU8::new(UNSET);

/// Environment variable that opts a *newly created* database into `DUP_FIXED` storage tables.
pub const ENV_VAR: &str = "RETH_STORAGE_DUPFIXED";

/// Returns whether the `DUP_FIXED` gate is enabled for this process.
///
/// On first call the gate is initialised from the [`ENV_VAR`] environment variable unless it was
/// already set programmatically via [`set_enabled`].
#[inline]
pub fn enabled() -> bool {
    match GATE.load(Ordering::Relaxed) {
        OFF => false,
        ON => true,
        _ => {
            let on = std::env::var(ENV_VAR).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            // Only initialise if still unset; otherwise respect a concurrent `set_enabled`.
            let _ = GATE.compare_exchange(
                UNSET,
                if on { ON } else { OFF },
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            GATE.load(Ordering::Relaxed) == ON
        }
    }
}

/// Overrides the `DUP_FIXED` gate for this process, bypassing the environment variable.
///
/// Primarily intended for tests. Must be called before creating the tables whose behaviour it
/// should affect.
pub fn set_enabled(on: bool) {
    GATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// Expands a compact `StorageEntry` encoding (`key[32] ++ minimal_be(value)`) in place into the
/// fixed 64-byte `DUP_FIXED` form (`key[32] ++ be32(value)`), left-padding the value with zeros.
///
/// No-op if the buffer is already 64 bytes. The buffer must contain a valid compact `StorageEntry`
/// encoding (at least 32 bytes, at most 64).
pub(crate) fn expand_buf(buf: &mut Vec<u8>) {
    debug_assert!(
        buf.len() >= KEY_LEN && buf.len() <= FIXED_WIDTH,
        "compact StorageEntry must be {KEY_LEN}..={FIXED_WIDTH} bytes, got {}",
        buf.len()
    );
    let value_len = buf.len() - KEY_LEN;
    if value_len == KEY_LEN {
        // Already 64 bytes: value occupies all 32 bytes.
        return;
    }
    let pad = KEY_LEN - value_len;
    // Grow to 64 bytes, then shift the value bytes to the end and zero the gap so the value is
    // stored as a full 32-byte big-endian integer.
    buf.resize(FIXED_WIDTH, 0);
    buf.copy_within(KEY_LEN..KEY_LEN + value_len, KEY_LEN + pad);
    for b in &mut buf[KEY_LEN..KEY_LEN + pad] {
        *b = 0;
    }
}

/// Contracts a fixed 64-byte `DUP_FIXED` `StorageEntry` encoding (`key[32] ++ be32(value)`) back
/// into the stock compact form (`key[32] ++ minimal_be(value)`), which stock `Decompress` can
/// decode.
///
/// If `bytes` is not exactly 64 bytes it is returned unchanged (already in compact form, e.g. a
/// table that is not stored `DUP_FIXED`).
pub(crate) fn contract(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if bytes.len() != FIXED_WIDTH {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let (key, value) = bytes.split_at(KEY_LEN);
    // Strip leading zero bytes of the big-endian value, matching `U256::to_compact`.
    let start = value.iter().position(|&b| b != 0).unwrap_or(value.len());
    let trimmed = &value[start..];
    let mut out = Vec::with_capacity(KEY_LEN + trimmed.len());
    out.extend_from_slice(key);
    out.extend_from_slice(trimmed);
    std::borrow::Cow::Owned(out)
}

/// Pads a subkey seek operand to the fixed record width for `DUP_FIXED` tables.
///
/// MDBX stores `DUP_FIXED` duplicates as fixed-size keys in a nested tree, so `GET_BOTH_RANGE`
/// (used by `seek_by_key_subkey`/`walk_dup`) requires the data operand to be exactly the fixed
/// width — passing the bare 32-byte subkey triggers an `MDBX_BAD_VALSIZE`/assert. We right-pad the
/// subkey with zero bytes, which is the lower bound (smallest value) for that subkey and therefore
/// positions the cursor at the first entry whose subkey is `>=` the requested one — identical to
/// the plain `DUP_SORT` prefix-seek semantics.
///
/// Returns the operand unchanged when the table is not `DUP_FIXED`/the gate is off, or when the
/// subkey is already at least the fixed width.
pub(crate) fn pad_seek_subkey<T: reth_db_api::table::Table>(
    subkey: &[u8],
) -> std::borrow::Cow<'_, [u8]> {
    if T::DUPFIXED && enabled() && subkey.len() < FIXED_WIDTH {
        let mut v = Vec::with_capacity(FIXED_WIDTH);
        v.extend_from_slice(subkey);
        v.resize(FIXED_WIDTH, 0);
        std::borrow::Cow::Owned(v)
    } else {
        std::borrow::Cow::Borrowed(subkey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{B256, U256};
    use reth_db_api::table::{Compress, Decompress};
    use reth_primitives_traits::StorageEntry;

    /// Encode a `StorageEntry` the way it would be written to a `DUP_FIXED` table: stock compress,
    /// then expand to fixed width.
    fn encode_fixed(entry: &StorageEntry) -> Vec<u8> {
        let mut buf = entry.compress();
        expand_buf(&mut buf);
        buf
    }

    /// Decode a fixed-width record back into a `StorageEntry`: contract, then stock decompress.
    fn decode_fixed(bytes: &[u8]) -> StorageEntry {
        let compact = contract(bytes);
        StorageEntry::decompress(&compact).expect("decompress")
    }

    fn roundtrip(entry: StorageEntry) {
        let encoded = encode_fixed(&entry);
        assert_eq!(encoded.len(), FIXED_WIDTH, "fixed encoding must be exactly 64 bytes");
        // Key is stored raw in the first 32 bytes.
        assert_eq!(&encoded[..32], entry.key.as_slice());
        // Value is stored as a full 32-byte big-endian integer.
        assert_eq!(&encoded[32..], &entry.value.to_be_bytes::<32>());
        let decoded = decode_fixed(&encoded);
        assert_eq!(decoded, entry, "round-trip must be lossless");
    }

    #[test]
    fn fixed_codec_roundtrip_edges() {
        let key = B256::repeat_byte(0xab);
        // zero value
        roundtrip(StorageEntry { key, value: U256::ZERO });
        // small value
        roundtrip(StorageEntry { key, value: U256::from(1u64) });
        roundtrip(StorageEntry { key, value: U256::from(0xffu64) });
        roundtrip(StorageEntry { key, value: U256::from(256u64) });
        // one full limb
        roundtrip(StorageEntry { key, value: U256::from(u64::MAX) });
        // max value
        roundtrip(StorageEntry { key, value: U256::MAX });
        // value with an interior zero byte to ensure we don't strip those
        roundtrip(StorageEntry { key: B256::ZERO, value: U256::from(0x0100u64) });
    }

    #[test]
    fn fixed_encoding_is_always_64_bytes() {
        let key = B256::random();
        for shift in 0..=255u32 {
            let value = if shift == 255 { U256::MAX } else { U256::from(1u64) << shift };
            let encoded = encode_fixed(&StorageEntry { key, value });
            assert_eq!(encoded.len(), FIXED_WIDTH, "encoding of 1<<{shift} must be 64 bytes");
            assert_eq!(decode_fixed(&encoded), StorageEntry { key, value });
        }
    }

    #[test]
    fn contract_passthrough_when_not_fixed_width() {
        // A byte string that is not 64 bytes long is returned unchanged.
        let compact = vec![7u8; 40];
        assert_eq!(contract(&compact).as_ref(), compact.as_slice());
    }
}
