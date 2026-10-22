from ohamlab_agent_lab.agents import AgentStore


def test_create_clone_and_report(tmp_path):
    store = AgentStore(tmp_path)
    first = store.create("Researcher", "researcher")
    clone = store.clone(first["id"], "Researcher Copy")
    assert first["id"] != clone["id"]
    assert clone["cloned_from"] == first["id"]
    report = store.save_report(first, "Find sources", "Found two sources")
    assert report.exists()
    assert "Found two sources" in report.read_text(encoding="utf-8")
