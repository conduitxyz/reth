//! Process-wide gate for the D1 "shadow flat" storage index.
//!
//! # Background
//!
//! reth stores each account's storage as one MDBX `DUP_SORT` sub-tree
//! ([`HashedStorages`](crate::tables::HashedStorages)) keyed by the hashed account. A cold `SLOAD`
//! is a `seek_by_key_subkey` descent into that account's sub-tree, which degrades
//! worse-than-linearly once one hot account accumulates millions of randomly-keyed slots.
//!
//! The shadow-flat design (Design 1a) keeps the `DUP_SORT` table canonical — the trie's
//! storage-root walk still iterates it, so state roots are byte-identical — and adds a second,
//! non-dup table [`HashedStoragesFlat`](crate::tables::HashedStoragesFlat) keyed by the 64-byte
//! composite [`HashedSlotKey`](crate::tables::HashedSlotKey)
//! (`hashed_address ++ hashed_slot`, big-endian). When the gate is on, every hashed-storage write
//! is mirrored into the flat table and the hot read
//! (`LatestStateProviderRef::hashed_storage_lookup`) becomes a single point `get` instead of a
//! dup-cursor descent.
//!
//! # Gating (opt-in, default OFF)
//!
//! The gate is a process-wide flag initialised from the [`ENV_VAR`] environment variable
//! (`RETH_STORAGE_FLAT=1` / `RETH_STORAGE_FLAT=true` turns it on) on first use, and can be
//! overridden programmatically via [`set_enabled`] (used by tests). Default is OFF, which is
//! byte-identical to stock reth: no flat writes, reads go through the `DUP_SORT` cursor.
//!
//! # Correctness
//!
//! With the gate on, the flat mirror is maintained to preserve DUP_SORT's absent-equals-zero
//! semantics: a slot is `put` into the flat table iff its value is non-zero, and `delete`d when it
//! becomes zero. A flat point lookup therefore returns the exact same value (or `None`) as the
//! canonical dup-cursor walk for every slot, including absent, zero and `U256::MAX`.

use std::sync::atomic::{AtomicU8, Ordering};

// Tri-state gate: 0 = uninitialised, 1 = off, 2 = on.
const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static GATE: AtomicU8 = AtomicU8::new(UNSET);

/// Environment variable that opts this process into the shadow flat storage index.
pub const ENV_VAR: &str = "RETH_STORAGE_FLAT";

/// Returns whether the shadow-flat gate is enabled for this process.
///
/// On first call the gate is initialised from the [`ENV_VAR`] environment variable unless it was
/// already set programmatically via [`set_enabled`].
#[inline]
pub fn enabled() -> bool {
    match GATE.load(Ordering::Relaxed) {
        OFF => false,
        ON => true,
        _ => {
            let on =
                std::env::var(ENV_VAR).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
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

/// Overrides the shadow-flat gate for this process, bypassing the environment variable.
///
/// Primarily intended for tests.
pub fn set_enabled(on: bool) {
    GATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}
