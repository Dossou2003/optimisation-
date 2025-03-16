import streamlit as st
import pandas as pd
from heuristic_with_dispatch import DispatchingExamScheduler, GeneticScheduler, SimulatedAnnealingScheduler
from heuristic_without_dispatch import SimpleScheduler, BaseExamScheduler
from exact_with_dispatch import ExactWithDispatch
from exact_without_dispatch import ExactWithoutDispatch

st.set_page_config(page_title="Planificateur d'Examens", layout="wide")

def main():
    st.title("Planificateur d'Examens")
    
    # Sidebar pour les configurations
    st.sidebar.header("Configuration")
    
    # Choix de la méthode
    method = st.sidebar.selectbox(
        "Choisir la méthode",
        ["Exacte avec dispatch", "Exacte sans dispatch", 
         "Heuristique avec dispatch", "Heuristique sans dispatch"]
    )
    
    # Configuration générale
    num_days = st.sidebar.number_input("Nombre de jours", min_value=1, value=3)
    num_slots = st.sidebar.number_input("Nombre de créneaux par jour", min_value=1, value=6)
    delta = st.sidebar.number_input("Temps de transition (delta)", min_value=0, value=1)

    # Initialisation des listes
    exams = []
    rooms = []

    # Interface principale divisée en colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration des Examens")
        
        # Upload des examens via CSV ou saisie manuelle
        upload_type = st.radio("Mode d'entrée des examens", ["Upload CSV", "Saisie manuelle"])
        
        if upload_type == "Upload CSV":
            uploaded_file = st.file_uploader("Uploader le fichier des examens (CSV)", type="csv")
            if uploaded_file:
                try:
                    exams_df = pd.read_csv(uploaded_file)
                    st.dataframe(exams_df)
                    # Conversion du DataFrame en liste d'examens
                    for _, row in exams_df.iterrows():
                        if "dispatch" in method.lower():
                            exam = {
                                'id': str(row['id']),
                                'duration': int(row['duration']),
                                'filieres': eval(row['filieres']) if isinstance(row['filieres'], str) else row['filieres']
                            }
                        else:
                            exam = {
                                'id': str(row['id']),
                                'duration': int(row['duration']),
                                'n_students': int(row['n_students'])
                            }
                        exams.append(exam)
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier CSV des examens: {str(e)}")
        else:
            # Interface de saisie manuelle des examens
            num_exams = st.number_input("Nombre d'examens", min_value=1, value=3)
            for i in range(num_exams):
                st.markdown(f"**Examen {i+1}**")
                exam_id = st.text_input(f"ID de l'examen {i+1}", value=f"Exam{i+1}", key=f"exam_id_{i}")
                duration = st.number_input(f"Durée de l'examen {i+1}", min_value=1, value=2, key=f"duration_{i}")
                
                if "dispatch" in method.lower():
                    num_filieres = st.number_input(f"Nombre de filières pour l'examen {i+1}", min_value=1, value=2, key=f"num_fil_{i}")
                    filieres = {}
                    for j in range(num_filieres):
                        filiere = st.text_input(f"Nom de la filière {j+1} (Examen {i+1})", value=f"F{j+1}", key=f"fil_{i}_{j}")
                        students = st.number_input(f"Nombre d'étudiants filière {j+1} (Examen {i+1})", min_value=1, value=30, key=f"stud_{i}_{j}")
                        filieres[filiere] = students
                    exams.append({'id': exam_id, 'duration': duration, 'filieres': filieres})
                else:
                    n_students = st.number_input(f"Nombre total d'étudiants pour l'examen {i+1}", min_value=1, value=50, key=f"n_stud_{i}")
                    exams.append({'id': exam_id, 'duration': duration, 'n_students': n_students})

    with col2:
        st.subheader("Configuration des Salles")
        
        upload_type_rooms = st.radio("Mode d'entrée des salles", ["Upload CSV", "Saisie manuelle"])
        
        if upload_type_rooms == "Upload CSV":
            uploaded_file_rooms = st.file_uploader("Uploader le fichier des salles (CSV)", type="csv")
            if uploaded_file_rooms:
                try:
                    rooms_df = pd.read_csv(uploaded_file_rooms)
                    st.dataframe(rooms_df)
                    # Conversion du DataFrame en liste de salles
                    for _, row in rooms_df.iterrows():
                        room = {
                            'id': str(row['id']),
                            'capacity': int(row['capacity']),
                            'availability': eval(row['availability']) if 'availability' in row else {}
                        }
                        rooms.append(room)
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier CSV des salles: {str(e)}")
        else:
            num_rooms = st.number_input("Nombre de salles", min_value=1, value=3)
            for i in range(num_rooms):
                st.markdown(f"**Salle {i+1}**")
                room_id = st.text_input(f"ID de la salle {i+1}", value=f"R{i+1}", key=f"room_{i}")
                capacity = st.number_input(f"Capacité de la salle {i+1}", min_value=1, value=50, key=f"cap_{i}")
                has_availability = st.checkbox(f"Définir des disponibilités pour la salle {i+1}", key=f"has_avail_{i}")
                
                if has_availability:
                    availability = {}
                    num_constraints = st.number_input(f"Nombre de contraintes pour la salle {i+1}", min_value=1, value=1, key=f"num_const_{i}")
                    for j in range(num_constraints):
                        day = st.number_input(f"Jour de la contrainte {j+1}", min_value=1, max_value=num_days, value=1, key=f"day_{i}_{j}")
                        slot = st.number_input(f"Créneau de la contrainte {j+1}", min_value=1, max_value=num_slots, value=1, key=f"slot_{i}_{j}")
                        availability[(day, slot)] = False
                    rooms.append({'id': room_id, 'capacity': capacity, 'availability': availability})
                else:
                    rooms.append({'id': room_id, 'capacity': capacity})

    # Vérification des données avant la planification
    if not exams:
        st.warning("Veuillez ajouter au moins un examen.")
    elif not rooms:
        st.warning("Veuillez ajouter au moins une salle.")
    else:
        # Bouton pour lancer la planification
        if st.button("Planifier les examens"):
            with st.spinner("Planification en cours..."):
                try:
                    if method == "Exacte avec dispatch":
                        scheduler = ExactWithDispatch({
                            "exams": exams,
                            "rooms": rooms,
                            "num_days": num_days,
                            "num_slots": num_slots,
                            "delta": delta
                        })
                    elif method == "Exacte sans dispatch":
                        scheduler = ExactWithoutDispatch({
                            "exams": exams,
                            "rooms": rooms,
                            "num_days": num_days,
                            "num_slots": num_slots
                        })
                    elif method == "Heuristique avec dispatch":
                        scheduler = DispatchingExamScheduler(exams, rooms, {}, num_days, num_slots, delta)
                    elif method == "Heuristique sans dispatch":
                        try:
                            # Pas besoin de conversion ici, la classe BaseExamScheduler s'en charge
                            scheduler = SimpleScheduler(
                                exams=exams,  # Passer directement les examens
                                rooms=rooms,
                                conflicts={},
                                num_days=num_days,
                                num_slots=num_slots,
                                delta=delta
                            )
                            
                            solution = scheduler.run()
                            if not solution:
                                st.error("Aucune solution trouvée")
                                return
                            return solution
                        except Exception as e:
                            st.error(f"Une erreur est survenue lors de la planification : {str(e)}")
                            return None

                    solution = scheduler.run()
                    
                    if solution:
                        st.success("Planification terminée !")
                        st.subheader("Résultats")
                        
                        if isinstance(solution, tuple):
                            solution = solution[0]
                        
                        planning_data = []
                        # Check if solution is a dictionary or list
                        if isinstance(solution, dict):
                            for exam_id, details in solution.items():
                                planning_data.append({
                                    'Examen': exam_id,
                                    'Jour': details['day'],
                                    'Créneau': details['start'],
                                    'Durée': details['duration'],
                                    'Salles': ', '.join(details.get('rooms', [])) if 'rooms' in details else 
                                             ', '.join(details.get('dispatching', {}).values())
                                })
                        elif isinstance(solution, list):
                            for exam in solution:
                                # Handle different solution formats
                                if 'examen' in exam:  # Format from exact_without_dispatch
                                    exam_id = exam['examen']
                                    # Find the duration from the original exams list
                                    exam_duration = next(e['duration'] for e in exams if e['id'] == exam_id)
                                    planning_data.append({
                                        'Examen': exam_id,
                                        'Jour': exam['jour'],
                                        'Créneau': exam['créneau'],
                                        'Durée': exam_duration,
                                        'Salles': ''  # Empty for exact solution without dispatch
                                    })
                                else:  # Format from other schedulers
                                    planning_data.append({
                                        'Examen': exam['id'],
                                        'Jour': exam['day'],
                                        'Créneau': exam['start'],
                                        'Durée': exam['duration'],
                                        'Salles': ', '.join(exam.get('rooms', [])) if 'rooms' in exam else 
                                                 ', '.join(exam.get('dispatching', {}).values())
                                    })
                        
                        if planning_data:
                            planning_df = pd.DataFrame(planning_data)
                            st.dataframe(planning_df)
                        else:
                            st.warning("Solution trouvée mais aucun examen n'a pu être planifié.")
                    else:
                        st.error("Aucune solution trouvée. Essayez de modifier les paramètres.")

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la planification : {str(e)}")

if __name__ == "__main__":
    main()