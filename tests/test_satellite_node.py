# pylint: disable=redefined-outer-name
"""
Unit tests for the SatelliteNode class.
"""
import asyncio
from unittest.mock import AsyncMock
import pytest
from src.satellite_node import SatelliteNode
from src.filter import ObservationRecord
from src.reputation import MAX_REPUTATION

@pytest.fixture
def mock_queue():
    """Fixture for a mocked asyncio.Queue."""
    return AsyncMock(spec=asyncio.Queue)

def test_satellite_node_init(mock_queue):
    """
    Test the initialization of a SatelliteNode.
    """
    node = SatelliteNode(node_id=5, queue=mock_queue)
    assert node.id == 5
    assert node.queue is mock_queue
    assert node.reputation == MAX_REPUTATION / 2
    assert node.exp_pos == 0
    assert node.performance_ema == 0.5
    assert node.local_dag is None
    assert node.sensor_data is None

def test_load_sensor_data(mock_queue):
    """
    Test that sensor data is loaded correctly.
    """
    node = SatelliteNode(node_id=1, queue=mock_queue)
    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2)

    assert node.sensor_data is None
    node.load_sensor_data(obs_record)
    assert node.sensor_data is obs_record
    assert node.sensor_data.nis == 2.0

@pytest.mark.asyncio
async def test_submit_transaction_no_data(mock_queue):
    """
    Test that submitting a transaction without sensor data raises a ValueError.
    """
    node = SatelliteNode(node_id=1, queue=mock_queue)
    with pytest.raises(ValueError, match="Satellite 1 has no sensor data loaded."):
        await node.submit_transaction(recipient_address=123)

@pytest.mark.asyncio
async def test_submit_transaction_success():
    """
    Test the successful submission of a transaction using a real queue.
    """
    # Arrange
    queue = asyncio.Queue() # Use a real queue instead of a mock
    node = SatelliteNode(node_id=1, queue=queue)
    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2)
    node.load_sensor_data(obs_record)

    # This task simulates the consumer (the DAG's listen loop)
    async def mock_dag_listener():
        _, _, future = await queue.get()
        # Simulate successful consensus by the DAG
        future.set_result((True, 2.0))
        queue.task_done()

    listener_task = asyncio.create_task(mock_dag_listener())

    # Act: This will put an item on the queue and wait for the future to be resolved
    result = await node.submit_transaction(recipient_address=123)

    # Assert
    assert result == (True, 2.0)

    # Wait for the listener to finish its work and check queue is empty
    await queue.join()
    assert queue.empty()
    # Ensure the listener task didn't have an exception
    await listener_task
