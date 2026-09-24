def generate_recommendations(
    attendance,
    hours_studied,
    previous_score,
    tutoring_sessions,
    motivation_level,
    physical_activity,
):
    """
    Generate personalized academic recommendations
    based on student inputs.
    """

    recommendations = []

    # Attendance
    if attendance < 75:
        recommendations.append(
            "Improve attendance to at least 75% "
            "to maintain consistent learning."
        )
    elif attendance < 85:
        recommendations.append(
            "Try to increase attendance above 85% "
            "for better learning consistency."
        )

    # Study hours
    if hours_studied < 2:
        recommendations.append(
            "Increase focused study time to at least "
            "2–3 hours per day."
        )
    elif hours_studied < 4:
        recommendations.append(
            "Consider adding another focused study "
            "session during the week."
        )

    # Previous score
    if previous_score < 60:
        recommendations.append(
            "Review foundational concepts from previous "
            "topics before moving to advanced material."
        )
    elif previous_score < 75:
        recommendations.append(
            "Use previous weak areas as a priority "
            "for revision and practice."
        )

    # Tutoring
    if tutoring_sessions == 0:
        recommendations.append(
            "Consider practice sessions or academic "
            "support for difficult topics."
        )

    # Motivation
    if str(motivation_level).lower() == "low":
        recommendations.append(
            "Break study goals into smaller achievable "
            "tasks and maintain a consistent routine."
        )

    # Physical activity
    if physical_activity < 2:
        recommendations.append(
            "Maintain regular physical activity alongside "
            "study sessions."
        )

    # Default
    if not recommendations:
        recommendations.append(
            "Maintain your current study habits and "
            "continue regular revision and practice."
        )

    return recommendations