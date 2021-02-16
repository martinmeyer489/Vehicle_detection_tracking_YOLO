-- Here are some example Queries to demonstrate how to work with the Database

-- Find out the average speed of an object
-- Were using a CTE (Common Table Expression) to maintain better overview

-- Filter for cars that are detected in the middle of the frame or calculate 
-- distance from pixel


SELECT DISTINCT object_id,
    class_label,
    ROUND((MAX(timestamp) - MIN(timestamp))/1000.0,2) as time_travelled_s,
    ROUND((MAX(c_x) - MIN(c_x))/18.0,2) as distance_travelled_m,
    ROUND(3.6*
    (((MAX(c_x) - MIN(c_x))/18.0)/
    ((MAX(timestamp) - MIN(timestamp))/1000.0)),2)
    as speed_kmh
FROM detections
-- Join with classes table (it does not really help in this case, 
-- as we only detect one class (cars) anyways
    LEFT JOIN classes 
    ON detections.class_id = classes.class_id
GROUP BY 1,2;