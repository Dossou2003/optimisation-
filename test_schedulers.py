from heuristic_with_dispatch import DispatchingExamScheduler, GeneticScheduler, SimulatedAnnealingScheduler
from heuristic_without_dispatch import SimpleScheduler, BaseExamScheduler
from exact_with_dispatch import ExactWithDispatch
from exact_without_dispatch import ExactWithoutDispatch

def test_without_dispatch():
    """Test des méthodes sans dispatch"""
    # Données d'exemple
    exams = [
        {'id': 'Exam1', 'duration': 2, 'n_students': 50},
        {'id': 'Exam2', 'duration': 3, 'n_students': 60},
        {'id': 'Exam3', 'duration': 2, 'n_students': 40}
    ]
    
    rooms = [
        {'id': 'R1', 'capacity': 30},
        {'id': 'R2', 'capacity': 40},
        {'id': 'R3', 'capacity': 50}
    ]

    num_days = 3
    num_slots = 6
    delta = 1

    # Test méthode exacte sans dispatch
    print("\n=== Test Exact Sans Dispatch ===")
    scheduler_exact = ExactWithoutDispatch({
        "exams": exams,
        "rooms": rooms,
        "num_days": num_days,
        "num_slots": num_slots
    })
    solution_exact = scheduler_exact.run()
    print("Solution exacte:", solution_exact)

    # Test heuristique simple sans dispatch
    print("\n=== Test Heuristique Simple Sans Dispatch ===")
    scheduler_simple = SimpleScheduler(
        exams=exams,
        rooms=rooms,
        conflicts={},
        num_days=num_days,
        num_slots=num_slots,
        delta=delta
    )
    solution_simple = scheduler_simple.run()
    print("Solution heuristique simple:", solution_simple)

def test_with_dispatch():
    """Test des méthodes avec dispatch"""
    # Données d'exemple
    exams = [
        {
            'id': 'Exam1',
            'duration': 2,
            'filieres': {'F1': 30, 'F2': 20}
        },
        {
            'id': 'Exam2',
            'duration': 3,
            'filieres': {'F1': 25, 'F2': 35}
        },
        {
            'id': 'Exam3',
            'duration': 2,
            'filieres': {'F1': 20, 'F2': 20}
        }
    ]
    
    rooms = [
        {'id': 'R1', 'capacity': 30},
        {'id': 'R2', 'capacity': 40},
        {'id': 'R3', 'capacity': 50}
    ]

    num_days = 3
    num_slots = 6
    delta = 1

    # Test méthode exacte avec dispatch
    print("\n=== Test Exact Avec Dispatch ===")
    scheduler_exact = ExactWithDispatch({
        "exams": exams,
        "rooms": rooms,
        "num_days": num_days,
        "num_slots": num_slots,
        "delta": delta
    })
    solution_exact = scheduler_exact.run()
    print("Solution exacte:", solution_exact)

    # Test heuristique avec dispatch
    print("\n=== Test Heuristique Avec Dispatch ===")
    scheduler_heuristic = DispatchingExamScheduler(
        exams=exams,
        rooms=rooms,
        conflicts={},
        num_days=num_days,
        num_slots=num_slots,
        delta=delta
    )
    solution_heuristic = scheduler_heuristic.run()
    print("Solution heuristique:", solution_heuristic)

    # Test algorithme génétique avec dispatch
    print("\n=== Test Algorithme Génétique Avec Dispatch ===")
    ga_scheduler = GeneticScheduler(
        scheduler_heuristic,
        population_size=20,
        generations=50
    )
    solution_ga = ga_scheduler.run()
    print("Solution génétique:", solution_ga)

    # Test recuit simulé avec dispatch
    print("\n=== Test Recuit Simulé Avec Dispatch ===")
    sa_scheduler = SimulatedAnnealingScheduler(
        scheduler_heuristic,
        initial_temp=1000,
        cooling_rate=0.95,
        iterations=1000
    )
    solution_sa = sa_scheduler.run()
    print("Solution recuit simulé:", solution_sa)

if __name__ == "__main__":
    print("=== Tests des méthodes sans dispatch ===")
    test_without_dispatch()
    
    print("\n=== Tests des méthodes avec dispatch ===")
    test_with_dispatch()