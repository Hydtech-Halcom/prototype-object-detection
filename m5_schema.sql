-- Module 5 - PostgreSQL tables
CREATE TABLE camera_status (
    camera_id TEXT PRIMARY KEY, online BOOLEAN, updated_at TIMESTAMP DEFAULT now());
CREATE TABLE detection_event (
    id SERIAL PRIMARY KEY, camera_id TEXT, class_name TEXT, confidence REAL,
    created_at TIMESTAMP DEFAULT now());
CREATE TABLE behavior_event (
    id SERIAL PRIMARY KEY, camera_id TEXT, rule TEXT, state TEXT,
    created_at TIMESTAMP DEFAULT now());
CREATE TABLE system_alarm (
    id SERIAL PRIMARY KEY, camera_id TEXT, message TEXT, created_at TIMESTAMP DEFAULT now());
CREATE TABLE user_login (
    id SERIAL PRIMARY KEY, username TEXT, login_at TIMESTAMP DEFAULT now());
