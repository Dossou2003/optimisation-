import random
import math
import copy
import time

# ------------------------------------------------------------
# Classe de base : BaseExamScheduler
# ------------------------------------------------------------
class BaseExamScheduler:
    def __init__(self, exams, rooms, conflicts, num_days, num_slots, delta):
        """
        exams : liste de dictionnaires, chacun de la forme
           { 'id': str, 'duration': int, 'n_students': int }
        rooms : liste de dictionnaires, par ex. { 'id': str, 'capacity': int, 'availability': dict (optionnel) }
        conflicts : dictionnaire { exam_id: set(exam_id en conflit) }
        num_days : nombre de jours d'examen (ensemble D)
        num_slots : nombre de créneaux horaires par jour (ensemble T)
        delta : temps minimal de transition entre examens dans une même salle
        """
        self.num_days = num_days
        self.num_slots = num_slots
        self.delta = delta
        self.exams = { exam['id']: exam for exam in exams }
        self.exam_ids = [ exam['id'] for exam in exams ]
        self.rooms = { room['id']: room for room in rooms }
        self.conflicts = conflicts

        # Initialisation des structures internes
        self.reset()
        
        # Métriques de performance
        self.scheduled_exams = 0
        self.execution_time = 0
        self.iterations = 0

    def reset(self):
        """Réinitialise les structures internes."""
        self.schedule = {}  # { exam_id: {'day': j, 'start': t, 'duration': d, 'rooms': [...] } }
        self.room_bookings = { room_id: {day: [] for day in range(1, self.num_days+1)}
                               for room_id in self.rooms }
        self.day_exam_schedule = { day: [] for day in range(1, self.num_days+1) }

    def is_room_available(self, room_id, day, start, duration):
        """
        Vérifie que la salle room_id est disponible le jour 'day' pour tous les créneaux de [start, start+duration-1].
        Si la salle dispose d'une disponibilité explicite, elle est vérifiée.
        Puis, on s'assure qu'aucune réservation existante (avec transition) ne bloque la période.
        """
        room = self.rooms[room_id]
        if 'availability' in room:
            for t in range(start, start+duration):
                if not room['availability'].get((day, t), True):
                    return False
        for (b_start, b_end) in self.room_bookings[room_id][day]:
            new_end = start + duration - 1
            if not (new_end + self.delta < b_start or start > b_end + self.delta):
                return False
        return True

    def book_room(self, room_id, day, start, duration):
        """
        Réserve la salle room_id le jour 'day' pour l'intervalle [start, start+duration-1].
        """
        new_end = start + duration - 1
        self.room_bookings[room_id][day].append((start, new_end))
        self.room_bookings[room_id][day].sort(key=lambda x: x[0])

    def is_exam_conflict_free(self, exam_id, day, start, duration):
        """
        Vérifie que, pour l'examen exam_id programmé le jour 'day' au créneau 'start' avec durée 'duration',
        aucun examen en conflit (selon conflicts) ne chevauche ce créneau.
        """
        new_end = start + duration - 1
        for (other_exam, other_start, other_duration) in self.day_exam_schedule[day]:
            if other_exam in self.conflicts.get(exam_id, set()):
                other_end = other_start + other_duration - 1
                if not (new_end < other_start or other_end < start):
                    return False
        return True

    def assign_rooms(self, day, start, exam):
        """
        Pour un examen candidate (défini par exam avec sa 'duration' et 'students'),
        retourne une affectation (liste de room_ids) telle que la somme des capacités couvre students
        et que pour chaque salle, la disponibilité est assurée sur l'intervalle [start, start+duration-1].
        Méthode gloutonne : on trie les salles disponibles par capacité décroissante.
        """
        needed = exam['students']  # Changé de 'n_students' à 'students'
        available_rooms = []
        for room_id in self.rooms:
            if self.is_room_available(room_id, day, start, exam['duration']):
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

    def schedule_exam(self, exam):
        """
        Tente de programmer l'examen (sans dispatching) pour un créneau admissible.
        L'examen est programmé s'il trouve un (jour, start) tel que :
          - La durée tient dans le jour.
          - Aucun examen en conflit n'est programmé sur ce créneau.
          - Un ensemble de salles disponibles peut couvrir n_students.
        Retourne True si programmé, sinon False.
        """
        exam_id = exam['id']
        duration = exam['duration']
        for day in range(1, self.num_days+1):
            for start in range(1, self.num_slots - duration + 2):
                if not self.is_exam_conflict_free(exam_id, day, start, duration):
                    continue
                rooms_assigned = self.assign_rooms(day, start, exam)
                if rooms_assigned is None:
                    continue
                self.schedule[exam_id] = {'day': day, 'start': start, 'duration': duration, 'rooms': rooms_assigned}
                for room_id in rooms_assigned:
                        self.book_room(room_id, day, start, duration)
                self.day_exam_schedule[day].append((exam_id, start, duration))
                return True
        return False

    def compute_objective(self):
        """
        Calcule T_start et T_end (créneau global = (day-1)*num_slots + t) et renvoie l'intervalle T_end - T_start.
        Cet indicateur mesure la compacité du planning.
        """
        T_start = float('inf')
        T_end = 0
        for exam_id, assign in self.schedule.items():
            day = assign['day']
            start = assign['start']
            duration = assign['duration']
            global_start = (day - 1) * self.num_slots + start
            global_end = (day - 1) * self.num_slots + (start + duration - 1)
            T_start = min(T_start, global_start)
            T_end = max(T_end, global_end)
        return T_end - T_start, T_start, T_end

    def compute_schedule_rate(self):
        """Retourne le taux d'examens programmés par rapport au total."""
        return len(self.schedule) / len(self.exam_ids) if self.exam_ids else 0

    def schedule_from_order(self, order):
        """
        À partir d'un ordre (liste de exam_id), tente de programmer tous les examens dans cet ordre.
        Retourne une copie de la solution (schedule) et l'objectif.
        Si un examen n'est pas programmé, une pénalité est appliquée.
        """
        self.reset()
        penalty = 1000
        unscheduled = 0
        for exam_id in order:
            exam = self.exams[exam_id]
            if not self.schedule_exam(exam):
                unscheduled += 1
        obj = self.compute_objective()[0] + unscheduled * penalty
        return copy.deepcopy(self.schedule), obj

    def display_schedule(self):
        """Affiche le planning obtenu et quelques indicateurs."""
        print("Planning des examens (sans dispatching) :")
        for exam_id, assign in self.schedule.items():
            rooms_str = ", ".join(assign['rooms'])
            print(f"  Examen {exam_id} -> Jour {assign['day']}, Créneau {assign['start']} (durée {assign['duration']}), Salles: {rooms_str}")
        obj, T_start, T_end = self.compute_objective()
        print(f"\nCompacité du planning (T_end - T_start) : {obj}")
        print(f"Taux d'examens programmés : {self.compute_schedule_rate():.2%}")

# ------------------------------------------------------------
# Heuristique 1 : SimpleScheduler (gloutonne)
# ------------------------------------------------------------
class SimpleScheduler(BaseExamScheduler):
    def __init__(self, exams, rooms, conflicts, num_days, num_slots, delta):
        super().__init__(exams, rooms, conflicts, num_days, num_slots, delta)

    def run(self):
        """Exécute la planification gloutonne simple sur l'ensemble des examens."""
        start_time = time.time()
        self.reset()
        exam_list = list(self.exams.values())
        
        # Tri des examens par nombre d'étudiants décroissant
        exam_list.sort(key=lambda x: x['n_students'], reverse=True)
        
        for exam in exam_list:
            self.schedule_exam(exam)
            
        self.execution_time = time.time() - start_time
        return self.schedule

    def schedule_exam(self, exam):
        """Tente de programmer l'examen pour un créneau admissible."""
        exam_id = exam['id']
        duration = exam['duration']
        n_students = exam['n_students']
        
        for day in range(1, self.num_days + 1):
            for start in range(1, self.num_slots - duration + 2):
                if not self.is_exam_conflict_free(exam_id, day, start, duration):
                    continue
                    
                # Trouver les salles disponibles
                available_rooms = []
                total_capacity = 0
                for room_id, room in self.rooms.items():
                    if self.is_room_available(room_id, day, start, duration):
                        available_rooms.append(room_id)
                        total_capacity += room['capacity']
                        if total_capacity >= n_students:
                            # On a assez de capacité
                            self.schedule[exam_id] = {
                                'day': day,
                                'start': start,
                                'duration': duration,
                                'rooms': available_rooms
                            }
                            # Marquer les salles comme occupées
                            for r_id in available_rooms:
                                self.book_room(r_id, day, start, duration)
                            return True
        return False

# ------------------------------------------------------------
# Heuristique 2 : Algorithme Génétique (GA)
# ------------------------------------------------------------
class GeneticScheduler:
    def __init__(self, base_scheduler, population_size=20, generations=50, crossover_rate=0.8, mutation_rate=0.2):
        """
        base_scheduler : instance de BaseExamScheduler
        """
        self.scheduler = base_scheduler
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time=0

    def initial_population(self):
        """Génère une population de permutations aléatoires."""
        pop = []
        for _ in range(self.population_size):
            perm = self.exam_ids[:]
            random.shuffle(perm)
            pop.append(perm)
        return pop

    def evaluate(self, order):
        """Évalue une permutation en appelant schedule_from_order et retourne l'objectif."""
        _, obj = self.scheduler.schedule_from_order(order)
        return obj

    def tournament_selection(self, population, fitness, k=3):
        """Sélection par tournoi."""
        selected = random.choice(population)
        for _ in range(k - 1):
            candidate = random.choice(population)
            if fitness[tuple(candidate)] < fitness[tuple(selected)]:
                selected = candidate
        return selected

    def crossover(self, parent1, parent2):
        """Order Crossover (OX) pour permutations."""
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
        """Mutation par échange de deux examens."""
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
        self.execution_time= time.time()- start_time
        return best_order, best_schedule, best_obj

# ------------------------------------------------------------
# Heuristique 3 : Recuit Simulé (SA)
# ------------------------------------------------------------
class SimulatedAnnealingScheduler:
    def __init__(self, base_scheduler, initial_temp=1000, cooling_rate=0.95, iterations=1000):
        self.scheduler = base_scheduler
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time=0

    def neighbor(self, order):
        """Génère un voisin par échange de deux examens."""
        new_order = order[:]
        a, b = random.sample(range(len(new_order)), 2)
        new_order[a], new_order[b] = new_order[b], new_order[a]
        return new_order

    def run(self):
        start_time=time.time()
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
        self.execution_time =time.time() - start_time
        return best_order, best_schedule, best_obj

# ------------------------------------------------------------
# Heuristique 4 : Recherche Tabou (TS)
# ------------------------------------------------------------
class TabuSearchScheduler:
    def __init__(self, base_scheduler, iterations=500, tabu_size=20):
        self.scheduler = base_scheduler
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.exam_ids = self.scheduler.exam_ids
        self.execution_time=0

    def neighbor(self, order):
        """Génère tous les voisins par échange de deux examens."""
        neighbors = []
        n = len(order)
        for i in range(n):
            for j in range(i+1, n):
                new_order = order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                neighbors.append(((i, j), new_order))
        return neighbors

    def run(self):
        start_time=time.time()
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
        self.execution_time=time.time() -start_time
        return best_order, best_schedule, best_obj

# ------------------------------------------------------------
# Exemple d'utilisation commune
# ------------------------------------------------------------
if __name__ == "__main__":
    # Définition des examens (sans dispatching)
    exams = [
        { 'id': 'Math', 'duration': 2, 'students': 100 },
        { 'id': 'Phys', 'duration': 2, 'students': 80 },
        { 'id': 'Info', 'duration': 3, 'students': 90 },
        { 'id': 'Chim', 'duration': 2, 'students': 70 },
        { 'id': 'Bio',  'duration': 1, 'students': 60 },
    ]
    # Conflits (exemple)
    conflict_list = [
        ('Math', 'Phys'),
        ('Info', 'Math'),
        ('Chim', 'Bio')
    ]
    conflicts = { exam['id']: set() for exam in exams }
    for e1, e2 in conflict_list:
        conflicts[e1].add(e2)
        conflicts[e2].add(e1)
    # Définition des salles
    rooms = [
        { 'id': 'R1', 'capacity': 50, 'availability': {(1,1): False} },
        { 'id': 'R2', 'capacity': 60 },
        { 'id': 'R3', 'capacity': 70 },
    ]
    num_days = 3
    num_slots = 6
    delta = 0  # pas de temps de transition dans cet exemple

    print("\n===== Heuristique Simple (gloutonne) =====")
    simple_scheduler = SimpleScheduler(exams, rooms, conflicts, num_days, num_slots, delta)
    simple_scheduler.run()
    simple_scheduler.display_schedule()
    print(f"Temps d'exécution (Simple) : {simple_scheduler.execution_time:.4f} secondes\n")

    # Pour les autres heuristiques, on utilisera la méthode schedule_from_order de la classe de base.
    base_scheduler = BaseExamScheduler(exams, rooms, conflicts, num_days, num_slots, delta)

    print("\n===== Algorithme Génétique =====")
    ga = GeneticScheduler(base_scheduler, population_size=20, generations=50)
    order_ga, schedule_ga, obj_ga = ga.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution (Simple) : {ga.execution_time:.4f} secondes\n")
    print("Objectif (T_end - T_start) =", obj_ga)

    print("\n===== Recuit Simulé =====")
    sa = SimulatedAnnealingScheduler(base_scheduler, initial_temp=1000, cooling_rate=0.95, iterations=1000)
    order_sa, schedule_sa, obj_sa = sa.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution (SA) : {sa.execution_time:.4f} secondes\n")
    print("Objectif (T_end - T_start) =", obj_sa)

    print("\n===== Recherche Tabou =====")
    ts = TabuSearchScheduler(base_scheduler, iterations=500, tabu_size=20)
    order_ts, schedule_ts, obj_ts = ts.run()
    base_scheduler.display_schedule()
    print(f"Temps d'exécution  : {ts.execution_time:.4f} secondes\n")
    print("Objectif (T_end - T_start) =", obj_ts)
