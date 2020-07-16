
from anabel import execute_model
from anabel.cli import main


def test_main():
    main([])


def test_execute_model():
    assert execute_model([b'a', b'bc', b'abc']) == b'abc'
