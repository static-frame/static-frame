from types import SimpleNamespace

from static_frame.core.store_client_mixin import StoreClientMixin


def test_store_client_mixin_resolve_config() -> None:
    expected = object()

    store = SimpleNamespace(_config=expected)
    bus = SimpleNamespace(_store=store)
    quilt = SimpleNamespace(_bus=bus)
    yarn = SimpleNamespace()
    nothing = SimpleNamespace()

    func = StoreClientMixin._resolve_config

    assert func(bus, config=None) is expected  # type: ignore
    assert func(quilt, config=None) is expected  # type: ignore
    assert func(yarn, config=None) is None  # type: ignore
    assert func(nothing, config=expected) is expected  # type: ignore
