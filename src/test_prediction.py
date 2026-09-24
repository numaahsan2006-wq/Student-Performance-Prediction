from prediction import predict_performance


def main():

    student = {
        "Hours_Studied": 25,
        "Attendance": 90,
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 2,
        "Family_Income": "Medium",
        "Teacher_Quality": "Good",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Female",
    }

    score, category = predict_performance(
        student
    )

    print("=" * 50)
    print("PREDICTION TEST")
    print("=" * 50)

    print(f"Predicted Score: {score:.2f}")
    print(f"Performance Level: {category}")


if __name__ == "__main__":
    main()