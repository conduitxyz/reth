use crate::{
    AccountReader, BlockHashReader, HashedPostStateProvider, StateProvider, StateRootProvider,
};
use alloy_primitives::{Address, BlockNumber, Bytes, StorageKey, StorageValue, B256};
use reth_db_api::{cursor::DbDupCursorRO, tables, tables::HashedSlotKey, transaction::DbTx};
use reth_primitives_traits::{Account, Bytecode};
use reth_storage_api::{
    BytecodeReader, DBProvider, StateProofProvider, StorageRootProvider, StorageSettingsCache,
};
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use reth_trie::{
    hashed_cursor::HashedPostStateCursorFactory,
    proof::{Proof, StorageProof},
    trie_cursor::InMemoryTrieCursorFactory,
    updates::TrieUpdates,
    witness::TrieWitness,
    AccountProof, ExecutionWitnessMode, HashedPostState, HashedStorage, KeccakKeyHasher,
    MultiProof, MultiProofTargets, StateRoot, StorageMultiProof, StorageRoot, TrieInput,
    TrieInputSorted,
};
use reth_trie_db::{DatabaseProof, DatabaseStateRoot, DatabaseStorageProof, DatabaseStorageRoot};

type DbStateRoot<'a, TX, A> = StateRoot<
    reth_trie_db::DatabaseTrieCursorFactory<&'a TX, A>,
    reth_trie_db::DatabaseHashedCursorFactory<&'a TX>,
>;
type DbStorageRoot<'a, TX, A> = StorageRoot<
    reth_trie_db::DatabaseTrieCursorFactory<&'a TX, A>,
    reth_trie_db::DatabaseHashedCursorFactory<&'a TX>,
>;
type DbStorageProof<'a, TX, A> = StorageProof<
    'static,
    reth_trie_db::DatabaseTrieCursorFactory<&'a TX, A>,
    reth_trie_db::DatabaseHashedCursorFactory<&'a TX>,
>;
type DbProof<'a, TX, A> = Proof<
    reth_trie_db::DatabaseTrieCursorFactory<&'a TX, A>,
    reth_trie_db::DatabaseHashedCursorFactory<&'a TX>,
>;
/// State provider over latest state that takes tx reference.
///
/// Wraps a [`DBProvider`] to get access to database.
#[derive(Debug)]
pub struct LatestStateProviderRef<'b, Provider>(&'b Provider);

impl<'b, Provider: DBProvider> LatestStateProviderRef<'b, Provider> {
    /// Create new state provider
    pub const fn new(provider: &'b Provider) -> Self {
        Self(provider)
    }

    fn tx(&self) -> &Provider::Tx {
        self.0.tx_ref()
    }

    fn hashed_storage_lookup(
        &self,
        hashed_address: B256,
        hashed_slot: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        // D1 shadow-flat read redirect: when the gate is on, serve the read from the flat
        // composite-key table as a single point lookup instead of a `DUP_SORT` sub-tree descent.
        // The flat table is kept in sync with `HashedStorages` by `write_hashed_state`, so the
        // returned value (or absence) is identical to the canonical dup-cursor walk.
        if reth_db_api::flat::enabled() {
            return Ok(self
                .tx()
                .get::<tables::HashedStoragesFlat>(HashedSlotKey::new(hashed_address, hashed_slot))?
                .map(|v| v.0));
        }

        let mut cursor = self.tx().cursor_dup_read::<tables::HashedStorages>()?;
        Ok(cursor
            .seek_by_key_subkey(hashed_address, hashed_slot)?
            .filter(|e| e.key == hashed_slot)
            .map(|e| e.value))
    }
}

impl<Provider: DBProvider + StorageSettingsCache> AccountReader
    for LatestStateProviderRef<'_, Provider>
{
    /// Get basic account information.
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        if self.0.cached_storage_settings().use_hashed_state() {
            let hashed_address = alloy_primitives::keccak256(address);
            self.tx()
                .get_by_encoded_key::<tables::HashedAccounts>(&hashed_address)
                .map_err(Into::into)
        } else {
            self.tx().get_by_encoded_key::<tables::PlainAccountState>(address).map_err(Into::into)
        }
    }
}

impl<Provider: BlockHashReader> BlockHashReader for LatestStateProviderRef<'_, Provider> {
    /// Get block hash by number.
    fn block_hash(&self, number: u64) -> ProviderResult<Option<B256>> {
        self.0.block_hash(number)
    }

    fn canonical_hashes_range(
        &self,
        start: BlockNumber,
        end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        self.0.canonical_hashes_range(start, end)
    }
}

impl<Provider: DBProvider + StorageSettingsCache> StateRootProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn state_root(&self, hashed_state: HashedPostState) -> ProviderResult<B256> {
        reth_trie_db::with_adapter!(self.0, |A| {
            let sorted = hashed_state.into_sorted();
            Ok(<DbStateRoot<'_, _, A> as DatabaseStateRoot<_>>::overlay_root(self.tx(), &sorted)?)
        })
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        reth_trie_db::with_adapter!(self.0, |A| {
            Ok(<DbStateRoot<'_, _, A> as DatabaseStateRoot<_>>::overlay_root_from_nodes(
                self.tx(),
                TrieInputSorted::from_unsorted(input),
            )?)
        })
    }

    fn state_root_with_updates(
        &self,
        hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        reth_trie_db::with_adapter!(self.0, |A| {
            let sorted = hashed_state.into_sorted();
            Ok(<DbStateRoot<'_, _, A> as DatabaseStateRoot<_>>::overlay_root_with_updates(
                self.tx(),
                &sorted,
            )?)
        })
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        reth_trie_db::with_adapter!(self.0, |A| {
            Ok(
                <DbStateRoot<'_, _, A> as DatabaseStateRoot<_>>::overlay_root_from_nodes_with_updates(
                    self.tx(),
                    TrieInputSorted::from_unsorted(input),
                )?,
            )
        })
    }
}

impl<Provider: DBProvider + StorageSettingsCache> StorageRootProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn storage_root(
        &self,
        address: Address,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<B256> {
        reth_trie_db::with_adapter!(self.0, |A| {
            <DbStorageRoot<'_, _, A>>::overlay_root(self.tx(), address, hashed_storage)
                .map_err(|err| ProviderError::Database(err.into()))
        })
    }

    fn storage_proof(
        &self,
        address: Address,
        slot: B256,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<reth_trie::StorageProof> {
        reth_trie_db::with_adapter!(self.0, |A| {
            <DbStorageProof<'_, _, A>>::overlay_storage_proof(
                self.tx(),
                address,
                slot,
                hashed_storage,
            )
            .map_err(ProviderError::from)
        })
    }

    fn storage_multiproof(
        &self,
        address: Address,
        slots: &[B256],
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        reth_trie_db::with_adapter!(self.0, |A| {
            <DbStorageProof<'_, _, A>>::overlay_storage_multiproof(
                self.tx(),
                address,
                slots,
                hashed_storage,
            )
            .map_err(ProviderError::from)
        })
    }
}

impl<Provider: DBProvider + StorageSettingsCache> StateProofProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn proof(
        &self,
        input: TrieInput,
        address: Address,
        slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        reth_trie_db::with_adapter!(self.0, |A| {
            let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(self.tx());
            proof.overlay_account_proof(input, address, slots).map_err(ProviderError::from)
        })
    }

    fn multiproof(
        &self,
        input: TrieInput,
        targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
        reth_trie_db::with_adapter!(self.0, |A| {
            let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(self.tx());
            proof.overlay_multiproof(input, targets).map_err(ProviderError::from)
        })
    }

    fn witness(
        &self,
        input: TrieInput,
        target: HashedPostState,
        mode: ExecutionWitnessMode,
    ) -> ProviderResult<Vec<Bytes>> {
        reth_trie_db::with_adapter!(self.0, |A| {
            let nodes_sorted = input.nodes.into_sorted();
            let state_sorted = input.state.into_sorted();
            let witness = TrieWitness::new(
                InMemoryTrieCursorFactory::new(
                    reth_trie_db::DatabaseTrieCursorFactory::<_, A>::new(self.tx()),
                    &nodes_sorted,
                ),
                HashedPostStateCursorFactory::new(
                    reth_trie_db::DatabaseHashedCursorFactory::new(self.tx()),
                    &state_sorted,
                ),
            )
            .with_prefix_sets_mut(input.prefix_sets)
            .with_execution_witness_mode(mode);
            let witness =
                if mode.is_canonical() { witness } else { witness.always_include_root_node() };
            let mut values: Vec<_> = witness.compute(target)?.into_values().collect();
            if mode.is_canonical() {
                values.sort_unstable();
            }
            Ok(values)
        })
    }
}

impl<Provider: DBProvider> HashedPostStateProvider for LatestStateProviderRef<'_, Provider> {
    fn hashed_post_state(&self, bundle_state: &revm_database::BundleState) -> HashedPostState {
        HashedPostState::from_bundle_state::<KeccakKeyHasher>(bundle_state.state())
    }
}

impl<Provider: DBProvider + BlockHashReader + StorageSettingsCache> StateProvider
    for LatestStateProviderRef<'_, Provider>
{
    /// Get storage by plain (unhashed) storage key slot.
    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        if self.0.cached_storage_settings().use_hashed_state() {
            self.hashed_storage_lookup(
                alloy_primitives::keccak256(account),
                alloy_primitives::keccak256(storage_key),
            )
        } else {
            let mut cursor = self.tx().cursor_dup_read::<tables::PlainStorageState>()?;
            if let Some(entry) = cursor.seek_by_key_subkey(account, storage_key)? &&
                entry.key == storage_key
            {
                return Ok(Some(entry.value));
            }
            Ok(None)
        }
    }
}

impl<Provider: DBProvider + BlockHashReader> BytecodeReader
    for LatestStateProviderRef<'_, Provider>
{
    /// Get account code by its hash
    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        self.tx().get_by_encoded_key::<tables::Bytecodes>(code_hash).map_err(Into::into)
    }
}

/// State provider for the latest state.
#[derive(Debug)]
pub struct LatestStateProvider<Provider>(Provider);

impl<Provider: DBProvider> LatestStateProvider<Provider> {
    /// Create new state provider
    pub const fn new(db: Provider) -> Self {
        Self(db)
    }

    /// Returns a new provider that takes the `TX` as reference
    #[inline(always)]
    const fn as_ref(&self) -> LatestStateProviderRef<'_, Provider> {
        LatestStateProviderRef::new(&self.0)
    }
}

// Delegates all provider impls to [LatestStateProviderRef]
reth_storage_api::macros::delegate_provider_impls!(LatestStateProvider<Provider> where [Provider: DBProvider + BlockHashReader + StorageSettingsCache]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_utils::create_test_provider_factory, StateWriter};
    use alloy_primitives::{address, b256, keccak256, U256};
    use reth_db_api::{
        cursor::DbDupCursorRO,
        flat,
        models::StorageSettings,
        tables,
        tables::HashedSlotKey,
        transaction::{DbTx, DbTxMut},
    };
    use reth_primitives_traits::StorageEntry;
    use reth_storage_api::StorageSettingsCache;
    use reth_trie::{HashedPostState, HashedStorage};

    const fn assert_state_provider<T: StateProvider>() {}
    #[expect(dead_code)]
    const fn assert_latest_state_provider<
        T: DBProvider + BlockHashReader + StorageSettingsCache,
    >() {
        assert_state_provider::<LatestStateProvider<T>>();
    }

    #[test]
    fn test_latest_storage_hashed_state() {
        let factory = create_test_provider_factory();
        factory.set_storage_settings_cache(StorageSettings::v2());

        let address = address!("0x0000000000000000000000000000000000000001");
        let slot = b256!("0x0000000000000000000000000000000000000000000000000000000000000001");

        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);

        let tx = factory.provider_rw().unwrap().into_tx();
        tx.put::<tables::HashedStorages>(
            hashed_address,
            StorageEntry { key: hashed_slot, value: U256::from(42) },
        )
        .unwrap();
        tx.commit().unwrap();

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);

        assert_eq!(provider_ref.storage(address, slot).unwrap(), Some(U256::from(42)));

        let other_address = address!("0x0000000000000000000000000000000000000099");
        let other_slot =
            b256!("0x0000000000000000000000000000000000000000000000000000000000000099");
        assert_eq!(provider_ref.storage(other_address, other_slot).unwrap(), None);

        let tx = factory.provider_rw().unwrap().into_tx();
        let plain_address = address!("0x0000000000000000000000000000000000000002");
        let plain_slot =
            b256!("0x0000000000000000000000000000000000000000000000000000000000000002");
        tx.put::<tables::PlainStorageState>(
            plain_address,
            StorageEntry { key: plain_slot, value: U256::from(99) },
        )
        .unwrap();
        tx.commit().unwrap();

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);
        assert_eq!(provider_ref.storage(plain_address, plain_slot).unwrap(), None);
    }

    #[test]
    fn test_latest_storage_hashed_state_returns_none_for_missing() {
        let factory = create_test_provider_factory();
        factory.set_storage_settings_cache(StorageSettings::v2());

        let address = address!("0x0000000000000000000000000000000000000001");
        let slot = b256!("0x0000000000000000000000000000000000000000000000000000000000000001");

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);
        assert_eq!(provider_ref.storage(address, slot).unwrap(), None);
    }

    #[test]
    fn test_latest_storage_legacy() {
        let factory = create_test_provider_factory();
        assert!(!factory.provider().unwrap().cached_storage_settings().use_hashed_state());

        let address = address!("0x0000000000000000000000000000000000000001");
        let slot = b256!("0x0000000000000000000000000000000000000000000000000000000000000005");

        let tx = factory.provider_rw().unwrap().into_tx();
        tx.put::<tables::PlainStorageState>(
            address,
            StorageEntry { key: slot, value: U256::from(42) },
        )
        .unwrap();
        tx.commit().unwrap();

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);

        assert_eq!(provider_ref.storage(address, slot).unwrap(), Some(U256::from(42)));

        let other_slot =
            b256!("0x0000000000000000000000000000000000000000000000000000000000000099");
        assert_eq!(provider_ref.storage(address, other_slot).unwrap(), None);
    }

    #[test]
    fn test_latest_storage_legacy_does_not_read_hashed() {
        let factory = create_test_provider_factory();
        assert!(!factory.provider().unwrap().cached_storage_settings().use_hashed_state());

        let address = address!("0x0000000000000000000000000000000000000001");
        let slot = b256!("0x0000000000000000000000000000000000000000000000000000000000000005");
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);

        let tx = factory.provider_rw().unwrap().into_tx();
        tx.put::<tables::HashedStorages>(
            hashed_address,
            StorageEntry { key: hashed_slot, value: U256::from(42) },
        )
        .unwrap();
        tx.commit().unwrap();

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);
        assert_eq!(provider_ref.storage(address, slot).unwrap(), None);
    }

    /// D1 shadow-flat equivalence test.
    ///
    /// With the gate ON, write many `(account, slot, value)` through the canonical writer
    /// ([`StateWriter::write_hashed_state`]) and assert that, for every slot (present, absent,
    /// zero, and `U256::MAX`), a read through the flat table returns the identical value as a
    /// direct `DUP_SORT` `seek_by_key_subkey` on the canonical [`tables::HashedStorages`] — and
    /// that `hashed_storage_lookup` (which redirects to the flat table under the gate) agrees
    /// with both.
    ///
    /// NOTE: the gate is process-wide; run under `nextest` (process-per-test isolation) so it does
    /// not leak into other tests. The gate is restored to OFF on exit as a best-effort for threaded
    /// `cargo test` runs.
    #[test]
    fn test_d1_shadow_flat_dupsort_equivalence() {
        flat::set_enabled(true);

        // Three accounts with overlapping slot hashes to exercise prefix grouping.
        let acct_a = b256!("0x1111111111111111111111111111111111111111111111111111111111111111");
        let acct_b = b256!("0x2222222222222222222222222222222222222222222222222222222222222222");
        let acct_c = b256!("0x3333333333333333333333333333333333333333333333333333333333333333");

        let slot = |n: u64| B256::from(U256::from(n));

        // (hashed_slot, value) per account. Includes zero (absent semantics) and U256::MAX.
        let data: Vec<(B256, Vec<(B256, U256)>)> = vec![
            (
                acct_a,
                vec![
                    (slot(1), U256::from(42u64)),
                    (slot(2), U256::MAX),
                    (slot(3), U256::ZERO), // zero => must be absent
                    (slot(1000), U256::from(1u64)),
                    (B256::repeat_byte(0xff), U256::from(0x0100u64)),
                ],
            ),
            (
                acct_b,
                vec![
                    (slot(1), U256::from(7u64)), // same slot hash as acct_a, different account
                    (slot(5), U256::from(u64::MAX)),
                    (slot(6), U256::ZERO),
                ],
            ),
            (acct_c, vec![(slot(9), U256::from(123456u64))]),
        ];

        run_equivalence(&data);

        flat::set_enabled(false);
    }

    /// Runs the write + triple-read equivalence check on a single fresh factory.
    fn run_equivalence(data: &[(B256, Vec<(B256, U256)>)]) {
        let factory = create_test_provider_factory();

        let provider_rw = factory.provider_rw().unwrap();
        let hashed_state = HashedPostState::default()
            .with_storages(
                data.iter()
                    .map(|(addr, slots)| {
                        (*addr, HashedStorage::from_iter(false, slots.iter().copied()))
                    })
                    .collect::<Vec<_>>(),
            )
            .into_sorted();
        provider_rw.write_hashed_state(&hashed_state).unwrap();
        provider_rw.commit().unwrap();

        let db = factory.provider().unwrap();
        let provider_ref = LatestStateProviderRef::new(&db);

        // Direct canonical DUP_SORT read (stock path, gate-independent).
        let dup_read = |a: B256, s: B256| -> Option<U256> {
            let mut cursor = db.tx_ref().cursor_dup_read::<tables::HashedStorages>().unwrap();
            cursor.seek_by_key_subkey(a, s).unwrap().filter(|e| e.key == s).map(|e| e.value)
        };
        // Direct flat point lookup (gate-independent).
        let flat_read = |a: B256, s: B256| -> Option<U256> {
            db.tx_ref()
                .get::<tables::HashedStoragesFlat>(HashedSlotKey::new(a, s))
                .unwrap()
                .map(|v| v.0)
        };

        for (addr, slots) in data {
            for (hashed_slot, value) in slots {
                let expected = if value.is_zero() { None } else { Some(*value) };
                let dup = dup_read(*addr, *hashed_slot);
                let flat_direct = flat_read(*addr, *hashed_slot);
                let lookup = provider_ref.hashed_storage_lookup(*addr, *hashed_slot).unwrap();

                assert_eq!(dup, expected, "dup mismatch for {addr:?}/{hashed_slot:?}");
                assert_eq!(flat_direct, dup, "flat vs dup mismatch for {addr:?}/{hashed_slot:?}");
                assert_eq!(
                    lookup, dup,
                    "hashed_storage_lookup (flat redirect) vs dup mismatch for {addr:?}/{hashed_slot:?}"
                );
            }

            // Absent slot for an existing account => None on both paths.
            let absent = B256::repeat_byte(0xab);
            assert_eq!(dup_read(*addr, absent), None);
            assert_eq!(flat_read(*addr, absent), None);
            assert_eq!(provider_ref.hashed_storage_lookup(*addr, absent).unwrap(), None);
        }

        // Entirely absent account => None on both paths.
        let ghost = B256::repeat_byte(0xee);
        let ghost_slot = B256::from(U256::from(1u64));
        assert_eq!(dup_read(ghost, ghost_slot), None);
        assert_eq!(flat_read(ghost, ghost_slot), None);
        assert_eq!(provider_ref.hashed_storage_lookup(ghost, ghost_slot).unwrap(), None);
    }
}
