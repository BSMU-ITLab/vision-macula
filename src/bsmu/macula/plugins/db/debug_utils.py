def print_sql_debug(query_text: str, query, values: list, eye_id_name: str, eye_id_value):
    """Выводит SQL-запрос, параметры и результат выполнения в консоль."""
    print("📤 SQL-запрос:")
    print(query_text)

    print("📦 Параметры:")
    for field in values:
        bound = query.boundValue(f":{field}")
        print(f"{field} → {bound}")
    print(f"{eye_id_name} → {eye_id_value}")
