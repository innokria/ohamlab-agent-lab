from ohamlab_agent_lab.models import AgentDefinition
from ohamlab_agent_lab.storage import FileStorage


def test_agent_round_trip(tmp_path):
    storage = FileStorage(tmp_path)
    agent = AgentDefinition(id="agent_1", name="Test Agent")
    storage.save_agent(agent)
    loaded = storage.get_agent("agent_1")
    assert loaded is not None
    assert loaded.name == "Test Agent"
