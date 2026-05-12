import sqlite3
try:
    conn = sqlite3.connect('trading.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET model_used = 'Sentiment_Transformer' WHERE model_used = 'Live Execution'")
    conn.commit()
    print("Updated past trades.")
finally:
    if conn:
        conn.close()