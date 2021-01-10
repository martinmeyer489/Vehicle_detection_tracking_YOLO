import sqlite3
import os
import sys

connection = sqlite3.connect("./database/test.db")

# Datensatz-Cursor erzeugen
cursor = connection.cursor()

# Datenbanktabelle erzeugen
sql = "CREATE TABLE personen(" \
      "name TEXT, " \
      "vorname TEXT, " \
      "personalnummer INTEGER PRIMARY KEY, " \
      "gehalt REAL, " \
      "geburtstag TEXT)"
cursor.execute(sql)

# Verbindung beenden
connection.close()