-- BrainModel.sql
-- A simple simulation of storing and retrieving information in neurons.

-- Drop table if it already exists
DROP TABLE IF EXISTS Neuron;

-- Create a table to represent a neuron
CREATE TABLE Neuron (
    id INT PRIMARY KEY,
    data TEXT
);

-- Encode information in the neuron (store sensory input)
INSERT INTO Neuron (id, data)
VALUES (1, 'This is some sensory input');

-- Retrieve the information stored in the neuron
SELECT data 
FROM Neuron
WHERE id = 1;
