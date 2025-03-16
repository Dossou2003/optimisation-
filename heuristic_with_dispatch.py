import copy
import random
import time
import math

# ============================================================
# Classe de base : DispatchingExamScheduler
# ============================================================
class DispatchingExamScheduler:
    def __init__(self, exams, rooms, conflicts, num_days, num_slots, delta):
        """
        Paramètres :
          - exams : liste de dictionnaires pour chaque examen, de la forme
                { 'id': str, 'duration': int, 'filieres': { filiere: n_students, ... } }
          - rooms : liste de dictionnaires, par ex. { 'id': str, 'capacity': int, 'availability': dict (optionnel) }
          - conflicts : dictionnaire { exam_id: set(exam_id en conflit) }
          - num_days : nombre de jours d'examen (ensemble D)
          - num_slots : nombre de créneaux horaires par jour (ensemble T)
          - delta : temps minimal de transition entre examens dans une même salle
        """
        self.exams = { exam['id']: exam for exam in exams }
        self.exam_ids = [ exam['id'] for exam in exams ]
        self.rooms = { room['id']: room for room in rooms }
        self.conflicts = conflicts
        self.num_days = num_days
        self.num_slots = num_slots
        self.delta = delta

        # Métriques
        self.execution_time = None
        self.schedule_rate = None
        self.objective_value = None
        self.solution = None  # Dictionnaire final de la solution

    def reset(self):
        """Réinitialise les structures internes pour une nouvelle solution."""
        self.schedule = {}  # { exam_id: { 'day': j, 'start': t, 'duration': d, 'dispatching': { filiere: room_id, ... } } }
        self.room_bookings = { room_id: {day: [] for day in range(1, self.num_days+1)}
                               for room_id in self.rooms }
        self.day_exam_schedule = { day: [] for day in range(1, self.num_days+1) }

    def is_room_available(self, room_id, day, start, duration, exam_id):
        """
        Vérifie que la salle room_id est disponible le jour 'day' pour l'intervalle [start, start+duration],
        en respectant la marge de transition (delta).
        """
        new_end = start + duration - 1
        if 'availability' in self.rooms[room_id]:
            for t in range(start, start+duration+1):
                if not self.rooms[room_id]['availability'].get((day, t), True):
                    return False
        for (b_start, b_end, b_exam, _) in self.room_bookings[room_id][day]:
            if b_exam == exam_id:
                continue
            if not (new_end + self.delta < b_start or start > b_end + self.delta):
                return False
        return True

    def book_room(self, room_id, day, start, duration, exam_id, added_capacity):
        """
        Réserve la salle room_id pour l'examen exam_id sur l'intervalle [start, start+duration-1].
        Si un booking pour exam_id existe déjà dans cette salle pour cet intervalle, on cumule la capacité.
        """
        new_end = start + duration - 1
        bookings = self.room_bookings[room_id][day]
        for i, (b_start, b_end, b_exam, used_cap) in enumerate(bookings):
            if b_exam == exam_id and b_start == start and b_end == new_end:
                bookings[i] = (b_start, b_end, b_exam, used_cap + added_capacity)
                return
        bookings.append((start, new_end, exam_id, added_capacity))
        bookings.sort(key=lambda x: x[0])
    
    def assign_rooms(self, exam, day, start):
        """
        Affecte un ensemble de salles pour l'examen, de sorte que la somme des capacités
        couvre le nombre total d'étudiants (calculé comme la somme des effectifs de toutes les filières).
        Méthode gloutonne : on trie les salles disponibles par capacité décroissante.
        Retourne la liste des salles affectées ou None si impossible.
        """
        needed = sum(exam['filieres'].values())
        available_rooms = []
        for room_id in self.rooms:
            if self.is_room_available(room_id, day, start, exam['duration'], exam['id']):
                cap = self.rooms[room_id]['capacity']
                available_rooms.append((room_id, cap))
        available_rooms.sort(key=lambda x: x[1], reverse=True)
        assigned = []
        total_cap = 0
        for room_id, cap in available_rooms:
            assigned.append(room_id)
            total_cap += cap
            if total_cap >= needed:
                return assigned
        return None

    def is_exam_conflict_free(self, exam_id, day, start, duration):
        """
        Vérifie qu'en programmant l'examen exam_id le jour 'day' à partir du créneau 'start' (durée duration),
        aucun examen en conflit ne chevauche ce créneau sur le même jour.
        """
        new_end = start + duration - 1
        for (other_exam, other_start, other_duration) in self.day_exam_schedule[day]:
            if other_exam in self.conflicts.get(exam_id, set()):
                other_end = other_start + other_duration - 1
                if not (new_end < other_start or other_end < start):
                    return False
        return True

    def schedule_exam(self, exam):
        """
        Tente de programmer l'examen avec dispatching pour un créneau admissible.
        Retourne True si l'examen est programmé, sinon False.
        """
        exam_id = exam['id']
        duration = exam['duration']
        for day in range(1, self.num_days + 1):
            for start in range(1, self.num_slots - duration + 2):
                if not self.is_exam_conflict_free(exam_id, day, start, duration):
                    continue
                rooms_assigned = self.assign_rooms(exam, day, start)
                if rooms_assigned is None:
                    continue
                # Affectation simple : pour chaque filière, on affecte la première salle disponible.
                dispatching = { f: rooms_assigned[0] for f in exam['filieres'] }
                self.schedule[exam_id] = {
                    'day': day,
                    'start': start,
                    'duration': duration,
                    'dispatching': dispatching
                }
                self.day_exam_schedule[day].append((exam_id, start, duration))
                for f in dispatching:
                    self.book_room(dispatching[f], day, start, duration, exam_id, exam['filieres'][f])
                return True
        print(f"Attention : l'examen {exam_id} n'a pas pu être programmé.")
        return False

    def schedule_exams(self):
        """
        Tente de programmer l'ensemble des examens.
        L'ordre de traitement est randomisé pour explorer différentes solutions.
        """
        self.reset()
        exam_list = list(self.exams.values())
        exam_list.sort(key=lambda exam: (sum(exam['filieres'].values()) + random.uniform(0, 0.1),
                                          len(self.conflicts.get(exam['id'], [])) + random.uniform(0, 0.1)),
                      reverse=True)
        for exam in exam_list:
            self.schedule_exam(exam)
    
    def iterative_greedy(self, iterations=10):
        """
        Exécute l'heuristique gloutonne de manière itérative en générant différents ordres aléatoires
        et retourne la meilleure solution trouvée ainsi que son objectif.
        """
        best_obj = float('inf')
        best_schedule = None
        for i in range(iterations):
            order = self.exam_ids[:]
            random.shuffle(order)
            sol, obj = self.schedule_from_order(order)
            print(f"Greedy Iteration {i+1}: Objectif = {obj}")
            if obj < best_obj:
                best_obj = obj
                best_schedule = copy.deepcopy(sol)
        return best_schedule, best_obj

    def compute_objective(self):
        """
        Calcule l'étalement global du planning en définissant :
          global_start = (day - 1) * num_slots + start.
        Retourne (T_end - T_start, T_start, T_end).
        """
        T_start_global = float('inf')
        T_end_global = 0
        for exam_id, assign in self.schedule.items():
            day = assign['day']
            start = assign['start']
            duration = assign['duration']
            global_start = (day - 1) * self.num_slots + start
            global_end = (day - 1) * self.num_slots + (start + duration - 1)
            T_start_global = min(T_start_global, global_start)
            T_end_global = max(T_end_global, global_end)
        return T_end_global - T_start_global, T_start_global, T_end_global

    def compute_schedule_rate(self):
        """Retourne le taux d'examens programmés par rapport au total."""
        return len(self.schedule) / len(self.exam_ids) if self.exam_ids else 0

    def schedule_from_order(self, order):
        """
        À partir d'un ordre (liste de exam_id), tente de programmer les examens dans cet ordre.
        Retourne une copie de la solution (schedule) et l'objectif (étalement global + pénalité pour examens non programmés).
        """
        self.reset()
        penalty = 1000  # Pénalité par examen non programmé (modifiable ou à supprimer)
        unscheduled = 0
        for exam_id in order:
            exam = self.exams[exam_id]
            if not self.schedule_exam(exam):
                unscheduled += 1
        obj = self.compute_objective()[0] + unscheduled * penalty
        return copy.deepcopy(self.schedule), obj

    def display_schedule(self):
        """Affiche le planning obtenu et quelques indicateurs."""
        if not self.schedule:
            print("Aucune solution disponible.")
        else:
            print("Planification des examens (dispatching intégré) :")
            for e, details in self.schedule.items():
                print(f"Examen {e} programmé le jour {details['day']} au créneau {details['start']} (durée {details['duration']})")
                disp_str = ", ".join([f"{filiere} -> {room}" for filiere, room in details['dispatching'].items()])
                print(f"  Dispatching: {disp_str}")
            obj, T_start, T_end = self.compute_objective()
            print(f"\nCompacité du planning (T_end - T_start) : {obj}")
            print(f"Taux d'examens programmés : {self.compute_schedule_rate() * 100:.2f}%")
    
    def run(self):
        start_time = time.time()
        order = self.exam_ids[:]
        random.shuffle(order)
        _, obj = self.schedule_from_order(order)
        self.solution, _ = self.schedule_from_order(order)
        self.schedule_rate = self.compute_schedule_rate()
        self.objective_value = obj
        self.execution_time = time.time() - start_time
        return self.solution

# ============================================================
# Heuristiques basées sur l'ordre de passage des examens
# ============================================================
class GeneticScheduler:
    def __init__(self, base_scheduler, population_size=20, generations=50, crossover_rate=0.8, mutation_rate=0.2):
        self.scheduler = base_scheduler
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time = 0

    def initial_population(self):
        pop = []
        for _ in range(self.population_size):
            perm = self.exam_ids[:]
            random.shuffle(perm)
            pop.append(perm)
        return pop

    def evaluate(self, order):
        _, obj = self.scheduler.schedule_from_order(order)
        return obj

    def tournament_selection(self, population, fitness, k=3):
        selected = random.choice(population)
        for _ in range(k - 1):
            candidate = random.choice(population)
            if fitness[tuple(candidate)] < fitness[tuple(selected)]:
                selected = candidate
        return selected

    def crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        a, b = sorted(random.sample(range(size), 2))
        child[a:b+1] = parent1[a:b+1]
        pos = (b+1) % size
        for gene in parent2:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    def mutate(self, order):
        a, b = random.sample(range(len(order)), 2)
        order[a], order[b] = order[b], order[a]
        return order

    def run(self):
        start_time = time.time()
        population = self.initial_population()
        best_order = None
        best_obj = float('inf')
        for gen in range(self.generations):
            fitness = { tuple(order): self.evaluate(order) for order in population }
            for order in population:
                f = fitness[tuple(order)]
                if f < best_obj:
                    best_obj = f
                    best_order = order[:]
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1[:]
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)
            population = new_population
            print(f"GA Gen {gen+1}: Meilleur objectif = {best_obj}")
        best_schedule, _ = self.scheduler.schedule_from_order(best_order)
        self.execution_time = time.time() - start_time
        return best_order, best_schedule, best_obj

class SimulatedAnnealingScheduler:
    def __init__(self, base_scheduler, initial_temp=1000, cooling_rate=0.95, iterations=1000):
        self.scheduler = base_scheduler
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time = 0

    def neighbor(self, order):
        new_order = order[:]
        a, b = random.sample(range(len(new_order)), 2)
        new_order[a], new_order[b] = new_order[b], new_order[a]
        return new_order

    def run(self):
        start_time = time.time()
        current_order = self.exam_ids[:]
        random.shuffle(current_order)
        _, current_obj = self.scheduler.schedule_from_order(current_order)
        best_order = current_order[:]
        best_obj = current_obj
        T = self.initial_temp
        for i in range(self.iterations):
            new_order = self.neighbor(current_order)
            _, new_obj = self.scheduler.schedule_from_order(new_order)
            delta = new_obj - current_obj
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_order = new_order
                current_obj = new_obj
                if current_obj < best_obj:
                    best_order = current_order[:]
                    best_obj = current_obj
            T *= self.cooling_rate
            if i % 100 == 0:
                print(f"SA Iter {i}: Temp = {T:.2f}, Best objective = {best_obj}")
        best_schedule, _ = self.scheduler.schedule_from_order(best_order)
        self.execution_time = time.time() - start_time
        return best_order, best_schedule, best_obj

class TabuSearchScheduler:
    def __init__(self, base_scheduler, iterations=500, tabu_size=20):
        self.scheduler = base_scheduler
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time = 0

    def neighbor(self, order):
        neighbors = []
        n = len(order)
        for i in range(n):
            for j in range(i+1, n):
                new_order = order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                neighbors.append(((i, j), new_order))
        return neighbors

    def run(self):
        start_time = time.time()
        current_order = self.exam_ids[:]
        random.shuffle(current_order)
        _, current_obj = self.scheduler.schedule_from_order(current_order)
        best_order = current_order[:]
        best_obj = current_obj
        tabu_list = []
        for iter in range(self.iterations):
            neighbs = self.neighbor(current_order)
            best_neighb = None
            best_neighb_obj = float('inf')
            best_move = None
            for move, order_candidate in neighbs:
                if move in tabu_list:
                    continue
                _, obj_candidate = self.scheduler.schedule_from_order(order_candidate)
                if obj_candidate < best_neighb_obj:
                    best_neighb_obj = obj_candidate
                    best_neighb = order_candidate
                    best_move = move
            if best_neighb is None:
                break
            current_order = best_neighb
            current_obj = best_neighb_obj
            tabu_list.append(best_move)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)
            if current_obj < best_obj:
                best_obj = current_obj
                best_order = current_order[:]
            if iter % 50 == 0:
                print(f"TS Iter {iter}: Best objective = {best_obj}")
        best_schedule, _ = self.scheduler.schedule_from_order(best_order)
        self.execution_time = time.time() - start_time
        return best_order, best_schedule, best_obj

# ============================================================
# Heuristique itérative gloutonne
# ============================================================
class IterativeGreedyScheduler:
    def __init__(self, base_scheduler, iterations=10, penalty=1000):
        self.scheduler = base_scheduler
        self.iterations = iterations
        self.penalty = penalty
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time = 0

    def run(self):
        start_time = time.time()
        best_obj = float('inf')
        best_schedule = None
        for i in range(self.iterations):
            order = self.exam_ids[:]
            random.shuffle(order)
            sol, obj = self.scheduler.schedule_from_order(order)
            print(f"Greedy Iteration {i+1}: Objectif = {obj}")
            if obj < best_obj:
                best_obj = obj
                best_schedule = copy.deepcopy(sol)
        self.execution_time = time.time() - start_time
        return best_schedule, best_obj

    def display_schedule(self):
        self.scheduler.display_schedule()

# ============================================================
# Exemple d'utilisation avec dispatching
# ============================================================
if __name__ == "__main__":
    exams = [
        { 'id': 'Math', 'duration': 2, 'filieres': { 'F1': 40, 'F2': 60 } },
        { 'id': 'Phys', 'duration': 2, 'filieres': { 'F1': 30, 'F2': 50 } },
        { 'id': 'Info', 'duration': 3, 'filieres': { 'F1': 45, 'F2': 45 } },
        { 'id': 'Chim', 'duration': 2, 'filieres': { 'F1': 35, 'F2': 35 } },
        { 'id': 'Bio',  'duration': 1, 'filieres': { 'F1': 25, 'F2': 35 } },
    ]
    
    conflicts = { exam['id']: set() for exam in exams }
    conflict_list = [
        ('Math', 'Phys'),
        ('Info', 'Math'),
        ('Chim', 'Bio')
    ]
    for e1, e2 in conflict_list:
        conflicts[e1].add(e2)
        conflicts[e2].add(e1)
    
    rooms = [
        {'id': 'R1', 'capacity': 50, 'availability': {(1,1): False}},
        {'id': 'R2', 'capacity': 60},
        {'id': 'R3', 'capacity': 70},
    ]
    
    num_days = 3
    num_slots = 6
    delta = 1
    
    print("\n===== Heuristique Simple (gloutonne) =====")
    simple_scheduler = DispatchingExamScheduler(exams, rooms, conflicts, num_days, num_slots, delta)
    simple_scheduler.schedule_exams()
    simple_scheduler.display_schedule()
    #print(f"Temps d'exécution (Simple) : {simple_scheduler.execution_time:.4f} secondes\n")
    
    base_scheduler = DispatchingExamScheduler(exams, rooms, conflicts, num_days, num_slots, delta)
    
    print("\n===== Algorithme Génétique =====")
    ga = GeneticScheduler(base_scheduler, population_size=20, generations=50)
    order_ga, schedule_ga, obj_ga = ga.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution (GA) : {ga.execution_time:.4f} secondes")
    print("Objectif (étalement global) =", obj_ga)
    
    print("\n===== Recuit Simulé =====")
    sa = SimulatedAnnealingScheduler(base_scheduler, initial_temp=1000, cooling_rate=0.95, iterations=1000)
    order_sa, schedule_sa, obj_sa = sa.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution (SA) : {sa.execution_time:.4f} secondes")
    print("Objectif (étalement global) =", obj_sa)
    
    print("\n===== Recherche Tabou =====")
    ts = TabuSearchScheduler(base_scheduler, iterations=500, tabu_size=20)
    order_ts, schedule_ts, obj_ts = ts.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution (TS) : {ts.execution_time:.4f} secondes")
    print("Objectif (étalement global) =", obj_ts)
    
    print("\n===== Iterative Greedy =====")
    ig = IterativeGreedyScheduler(base_scheduler, iterations=10)
    schedule_ig, obj_ig = ig.run()
    ig.display_schedule()
    print(f"Temps d'exécution (Iterative Greedy) : {ig.execution_time:.4f} secondes")
    print("Objectif (étalement global) =", obj_ig)
