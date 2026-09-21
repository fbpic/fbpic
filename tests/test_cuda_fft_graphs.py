from fbpic.fields.spectral_transform.fourier import _CudaGraphCache


class RecordingGraph:
    def __init__(self, events):
        self.events = events

    def launch(self, stream):
        self.events.append(("launch", stream))


class RecordingStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("exit")

    def begin_capture(self):
        self.events.append("begin_capture")

    def end_capture(self):
        self.events.append("end_capture")
        return RecordingGraph(self.events)


def test_cuda_graph_cache_warms_captures_then_replays():
    events = []
    stream = RecordingStream(events)
    replay_stream = object()
    cache = _CudaGraphCache(stream, replay_stream)

    def operation():
        events.append("operation")

    cache.run(("forward", 100, 200), operation)
    cache.run(("forward", 100, 200), operation)
    cache.run(("forward", 100, 200), operation)

    assert events == [
        "operation",
        "enter",
        "begin_capture",
        "operation",
        "end_capture",
        ("launch", replay_stream),
        "exit",
        ("launch", replay_stream),
    ]


def test_cuda_graph_cache_tracks_device_pointer_pairs_independently():
    events = []
    cache = _CudaGraphCache(RecordingStream(events), object())

    cache.run(("forward", 100, 200), lambda: events.append("first"))
    cache.run(("forward", 300, 400), lambda: events.append("second"))

    assert events == ["first", "second"]
