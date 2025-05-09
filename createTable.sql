CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    role INT NOT NULL DEFAULT 0,
    created_time TIMESTAMP
);

CREATE TABLE equipment_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    model VARCHAR(50),
    supplier VARCHAR(100)
);

CREATE TABLE equipment_status (
    id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    type VARCHAR(50) NOT NULL,
    level INT NOT NULL,
    location VARCHAR(100) NOT NULL,
    description VARCHAR(255),
    handler INT,
    FOREIGN KEY (equipment_id) REFERENCES equipment_info(id),
    FOREIGN KEY (handler) REFERENCES user(id)
);

CREATE TABLE device_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    model VARCHAR(50) NOT NULL,
    equipment_id INT NOT NULL,
    location VARCHAR(255) NOT NULL,
    created_time DATETIME NOT NULL,
    FOREIGN KEY (equipment_id) REFERENCES equipment_info(id)
);

CREATE TABLE model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(100) NOT NULL,
    version VARCHAR(10) NOT NULL,
    created_time DATETIME NOT NULL,
    created_by INT NOT NULL,
    description VARCHAR(255),
    FOREIGN KEY (created_by) REFERENCES user(id)
);