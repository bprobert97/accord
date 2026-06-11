"""
Unit tests for the Transaction class.
"""
from src.transaction import Transaction, TransactionMetadata, \
    TransactionAddresses

def test_transaction_creation_and_hash() -> None:
    """
    Test that a transaction is created correctly and its hash is consistent.
    """
    metadata = TransactionMetadata()
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="test_key",)

    # Fixed: Passed parent_hashes=() to satisfy constructor constraints
    tx = Transaction(
        addresses=addresses,
        tx_data="test_data",
        metadata=metadata,
        parent_hashes=()
    )

    # Hash should be a non-empty string
    assert isinstance(tx.hash, str)
    assert len(tx.hash) == 64  # SHA-256

    # Creating the exact same transaction should result in the same hash
    timestamp = metadata.timestamp
    tx2_metadata = TransactionMetadata(timestamp=timestamp)

    # Fixed: Passed matching empty parent hashes to yield an identical signature
    tx2 = Transaction(
        addresses=addresses,
        tx_data="test_data",
        metadata=tx2_metadata,
        parent_hashes=()
    )
    assert tx.hash == tx2.hash

    # Changing payload data should result in a different hash
    tx3_metadata = TransactionMetadata(timestamp=timestamp)
    tx3 = Transaction(
        addresses=TransactionAddresses(sender_address=99,
                                       recipient_address=2,
                                       sender_private_key="test_key",),
        tx_data="test_data",
        metadata=tx3_metadata,
        parent_hashes=()
    )
    assert tx.hash != tx3.hash

    # Security Verification: Changing graph parent hashes must yield a different hash
    tx_diff_parents = Transaction(
        addresses=addresses,
        tx_data="test_data",
        metadata=tx2_metadata,
        parent_hashes=("Genesis Transaction 1", "Genesis Transaction 2")
    )
    assert tx.hash != tx_diff_parents.hash


def test_transaction_repr() -> None:
    """
    Test the __repr__ method of the Transaction class.
    """
    metadata = TransactionMetadata()
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="test_key",)

    # Fixed: Passed parent_hashes=() to satisfy constructor constraints
    tx = Transaction(
        addresses=addresses,
        tx_data="test_data",
        metadata=metadata,
        parent_hashes=()
    )

    repr_str = repr(tx)
    assert isinstance(repr_str, str)
    assert "Transaction(" in repr_str
    assert "sender_address=1" in repr_str
    assert "tx_data='test_data'" in repr_str
    assert f"hash={tx.hash[:10]}..." in repr_str
