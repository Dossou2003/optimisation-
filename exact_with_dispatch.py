from ortools.sat.python import cp_model

class ExactWithDispatch:
    def __init__(self, data):
        self.exams = data["exams"]
        self.rooms = data["rooms"]
        self.num_days = data["num_days"]
        self.num_slots = data["num_slots"]
        self.delta = data["delta"]
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Variables de décision
        self.y = {}  # y[e,j,t] = 1 si l'examen e est programmé le jour j au créneau t
        self.x = {}  # x[e,r,j,t] = 1 si l'examen e utilise la salle r le jour j au créneau t
        self.z = {}  # z[e,f,r,j,t] = 1 si la filière f de l'examen e est dans la salle r le jour j au créneau t

    def build_model(self):
        # Création des variables
        for e in self.exams:
            for j in range(self.num_days):
                for t in range(self.num_slots):
                    self.y[e['id'], j, t] = self.model.NewBoolVar(f'y_{e["id"]}_{j}_{t}')
                    for r in self.rooms:
                        self.x[e['id'], r['id'], j, t] = self.model.NewBoolVar(f'x_{e["id"]}_{r["id"]}_{j}_{t}')
                        for f in e.get('filieres', {}).keys():
                            self.z[e['id'], f, r['id'], j, t] = self.model.NewBoolVar(f'z_{e["id"]}_{f}_{r["id"]}_{j}_{t}')

        # Contraintes
        for e in self.exams:
            # Chaque examen doit être programmé exactement une fois
            self.model.Add(sum(self.y[e['id'], j, t] 
                             for j in range(self.num_days) 
                             for t in range(self.num_slots)) == 1)

        # Objectif : minimiser la durée totale
        T_start = self.model.NewIntVar(0, self.num_days * self.num_slots, 'T_start')
        T_end = self.model.NewIntVar(0, self.num_days * self.num_slots, 'T_end')

        for e in self.exams:
            for j in range(self.num_days):
                for t in range(self.num_slots):
                    self.model.Add(T_start <= (j * self.num_slots + t) * self.y[e['id'], j, t])
                    self.model.Add(T_end >= (j * self.num_slots + t + e['duration'] - 1) * self.y[e['id'], j, t])

        self.model.Minimize(T_end - T_start)

    def solve(self):
        self.build_model()
        status = self.solver.Solve(self.model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self.get_solution(), self.solver.NumBooleans(), self.solver.ObjectiveValue()
        return None, 0, float('inf')

    def get_solution(self):
        solution = []
        for e in self.exams:
            for j in range(self.num_days):
                for t in range(self.num_slots):
                    if self.solver.Value(self.y[e["id"], j, t]):
                        rooms = []
                        for r in self.rooms:
                            if self.solver.Value(self.x[e["id"], r["id"], j, t]):
                                rooms.append(r["id"])
                        solution.append({
                            "examen": e["id"],
                            "jour": j + 1,
                            "créneau": t + 1,
                            "salles": rooms
                        })
        return solution

    def run(self):
        return self.solve()
