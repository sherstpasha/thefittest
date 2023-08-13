from thefittest.base._ea import LastBest


def test_lastbest():
    lastbest = LastBest()

    counter_1 = lastbest._no_increase_counter

    lastbest._update(1)
    counter_2 = lastbest._no_increase_counter

    lastbest._update(1)
    counter_3 = lastbest._no_increase_counter
    lastbest._update(2)
    counter_4 = lastbest._no_increase_counter

    assert counter_1 == counter_2
    assert counter_2 == counter_3 - 1
    assert counter_4 == 0
    


