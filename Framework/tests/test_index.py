from src.index import SimpleVectorIndex


def test_add_and_search():
    idx = SimpleVectorIndex()
    idx.add('a', [1, 0, 0], {'text': 'a'})
    idx.add('b', [0, 1, 0], {'text': 'b'})
    idx.add('c', [0, 0, 1], {'text': 'c'})

    res = idx.search([1, 0, 0], k=1)
    assert res[0][0] == 'a'


if __name__ == '__main__':
    test_add_and_search()
    print('test_index: OK')
