//! Small database table utilities and helper functions.

use super::dupfixed;
use crate::{
    table::{Decode, Decompress, Table, TableRow},
    DatabaseError,
};
use std::borrow::Cow;

/// Decompresses a value, first contracting it from fixed-width form when the table is stored
/// `DUP_FIXED` and the gate is enabled.
fn decompress_value<T>(v: Cow<'_, [u8]>) -> Result<T::Value, DatabaseError>
where
    T: Table,
    T::Value: Decompress,
{
    if T::DUPFIXED && dupfixed::enabled() {
        let contracted = dupfixed::contract(v.as_ref());
        return Ok(T::Value::decompress(contracted.as_ref())?);
    }
    Ok(match v {
        Cow::Borrowed(v) => Decompress::decompress(v)?,
        Cow::Owned(v) => Decompress::decompress_owned(v)?,
    })
}

/// Helper function to decode a `(key, value)` pair.
pub(crate) fn decoder<'a, T>(
    (k, v): (Cow<'a, [u8]>, Cow<'a, [u8]>),
) -> Result<TableRow<T>, DatabaseError>
where
    T: Table,
    T::Key: Decode,
    T::Value: Decompress,
{
    Ok((
        match k {
            Cow::Borrowed(k) => Decode::decode(k)?,
            Cow::Owned(k) => Decode::decode_owned(k)?,
        },
        decompress_value::<T>(v)?,
    ))
}

/// Helper function to decode only a value from a `(key, value)` pair.
pub(crate) fn decode_value<'a, T>(
    kv: (Cow<'a, [u8]>, Cow<'a, [u8]>),
) -> Result<T::Value, DatabaseError>
where
    T: Table,
{
    decompress_value::<T>(kv.1)
}

/// Helper function to decode a value. It can be a key or subkey.
pub(crate) fn decode_one<T>(value: Cow<'_, [u8]>) -> Result<T::Value, DatabaseError>
where
    T: Table,
{
    decompress_value::<T>(value)
}
