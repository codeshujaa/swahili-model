# Swahili Hate Speech Detection Model
A deep learning bert model for detecting hate speech in Swahili text, deployed as a Flask web application with Docker support.

This project implements a hate speech detection system specifically designed for the Swahili language. The model can classify Swahili text as either containing hate speech or being neutral, helping to moderate content and promote safer online environments in Swahili-speaking communities.

## Features

Swahili Language Support: Specifically trained and optimized for Swahili text
REST API: Easy-to-use Flask API for integration
Docker Support: Containerized deployment for easy scaling
Real-time Detection: Fast inference for live content moderation
High Accuracy: Trained on curated Swahili hate speech datasets


## Clone the repository

` git clone https://github.com/codeshujaa/swahili-model.git `

`cd swahili-model`

## Install dependencies

`pip install -r requirements.txt`


## Build the Docker image

`docker build -t swahili-hate-speech .`

## Run the container
`docker run -p 5000:5000 swahili-hate-speech`
