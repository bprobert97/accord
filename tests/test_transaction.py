"""
Unit tests for the Transaction class.
"""
from src.transaction import Transaction, TransactionMetadata

def test_transaction_creation_and_hash():
    """
    Test that a transaction is created correctly and its hash is consistent.
    """
    metadata = TransactionMetadata()
    tx = Transaction(
        sender_address=1,
        recipient_address=2,
        sender_private_key="test_key",
        tx_data="test_data",
        metadata=metadata
    )

    # Hash should be a non-empty string
    assert isinstance(tx.hash, str)
    assert len(tx.hash) == 64 # SHA-256

    # Creating the exact same transaction should result in the same hash
    # We need to control the timestamp for this
    timestamp = metadata.timestamp

    tx2_metadata = TransactionMetadata(timestamp=timestamp)
    tx2 = Transaction(
        sender_address=1,
        recipient_address=2,
        sender_private_key="test_key",
        tx_data="test_data",
        metadata=tx2_metadata
    )
    assert tx.hash == tx2.hash

    # Changing any data should result in a different hash
    tx3_metadata = TransactionMetadata(timestamp=timestamp)
    tx3 = Transaction(
        sender_address=99, # Changed sender
        recipient_address=2,
        sender_private_key="test_key",
        tx_data="test_data",
        metadata=tx3_metadata
    )
    assert tx.hash != tx3.hash

def test_transaction_repr():
    """
    Test the __repr__ method of the Transaction class.
    """
    metadata = TransactionMetadata()
    tx = Transaction(
        sender_address=1,
        recipient_address=2,
        sender_private_key="test_key",
        tx_data="test_data",
        metadata=metadata
    )

    repr_str = repr(tx)
    assert isinstance(repr_str, str)
    assert "Transaction(" in repr_str
    assert "sender_address=1" in repr_str
    assert "tx_data='test_data'" in repr_str
    assert f"hash={tx.hash[:10]}..." in repr_str
