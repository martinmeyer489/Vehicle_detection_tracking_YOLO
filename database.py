import sqlite3
from sqlite3 import Error

def insert_detections(conn, ):
    sql_insert_detection = """
                            INSERT INTO detections
                            [], 

                            """


def init_db(conn):
    """ create the tables etc, if they dont already exist
    """ 

    sql_create_detections_table = """CREATE TABLE IF NOT EXISTS detections (
                                    timestamp integer NOT NULL,
                                    object_id integer NOT NULL,
                                    c_x integer NOT NULL,
                                    c_y integer NOT NULL,
                                    bb_w integer,
                                    bb_h integer,
                                    object_type text,
                                    confidence double,
                                    missing_detection boolean,
                                    inserted_db_timestamp integer NOT NULL
                                    PRIMARY KEY (timestamp, object_id)
                                );"""

    create_table(conn, sql_create_detections_table)


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        init_db(conn) 
        return conn
    except Error as e:
        print(e)
    
    return conn


if __name__ == '__main__':
    create_connection(r"database.db")