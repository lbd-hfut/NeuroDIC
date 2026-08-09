"""Python thin-package import test."""


def test_import_neurodic() -> None:
    import neurodic

    assert callable(neurodic.pin_dic)
    assert callable(neurodic.pin_multi_slover_dic)
    assert callable(neurodic.pin_multi_pair_roi)
    assert callable(neurodic.ndef_dic)
    assert callable(neurodic.calibrate)
    assert isinstance(neurodic.native_available(), bool)
