from ortools.sat.python import cp_model

class ExactWithoutDispatch:
    def __init__(self, data):
        self.exams = data["exams"]
        self.rooms = data["rooms"]
        self.num_days = data["num_days"]
        self.num_slots = data["num_slots"]
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # Stocker les variables pour un accès dans get_solution()
        self.y = {}
        self.x = {}

    def build_model(self):
        exams = self.exams
        rooms = self.rooms
        num_days = self.num_days
        num_slots = self.num_slots
        model = self.model

        # Déclaration des variables de décision
        for e in exams:
            # Convertir n_students en students si nécessaire
            if 'n_students' in e:
                e['students'] = e['n_students']
                
            for j in range(num_days):
                for t in range(num_slots):
                    self.y[e['id'], j, t] = model.NewBoolVar(f'y_{e["id"]}_{j}_{t}')
                    for r in rooms:
                        self.x[e['id'], r['id'], j, t] = model.NewBoolVar(f'x_{e["id"]}_{r["id"]}_{j}_{t}')

        # Chaque examen doit être programmé exactement une fois
        for e in exams:
            model.Add(sum(self.y[e['id'], j, t] for j in range(num_days) for t in range(num_slots)) == 1)

        # La somme des capacités des salles doit couvrir le nombre d'étudiants
        for e in exams:
            students_count = e.get('students', e.get('n_students', 0))  # Gestion des deux formats possibles
            for j in range(num_days):
                for t in range(num_slots):
                    model.Add(sum(self.x[e['id'], r['id'], j, t] * r['capacity'] for r in rooms) >= 
                            students_count * self.y[e['id'], j, t])

        # Non-chevauchement des examens dans une même salle
        for r in rooms:
            for j in range(num_days):
                for t in range(num_slots):
                    model.Add(sum(self.x[e['id'], r['id'], j, tau] for e in exams for tau in range(t, t + e['duration']+1) if tau < num_slots) <= 1)

        # Disponibilité des salles
        for r in rooms:
            if 'availability' in r:
                for (j, t), available in r['availability'].items():
                    if not available:
                        for e in exams:
                            model.Add(self.x[e['id'], r['id'], j, t] == 0)

        # Objectif : minimiser la durée totale de la période d'examens
        T_start = model.NewIntVar(0, num_days * num_slots, 'T_start')
        T_end = model.NewIntVar(0, num_days * num_slots, 'T_end')

        for e in exams:
            for j in range(num_days):
                for t in range(num_slots):
                    model.Add(T_start <= (j * num_slots + t) * self.y[e['id'], j, t])
                    model.Add(T_end >= (j * num_slots + t + e['duration'] - 1) * self.y[e['id'], j, t])

        model.Minimize(T_end - T_start)

    def solve(self):
        self.build_model()
        status = self.solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self.get_solution(), self.solver.NumBooleans(), self.solver.ObjectiveValue()
        else:
            return None, 0, float('inf')

    def get_solution(self):
        solution = []
        for e in self.exams:
            for j in range(self.num_days):
                for t in range(self.num_slots):
                    if self.solver.Value(self.y[e["id"], j, t]):  # Correction ici
                        solution.append({
                            "examen": e["id"],
                            "jour": j + 1,
                            "créneau": t + 1
                        })
        return solution

    def run(self):
        return self.solve()
