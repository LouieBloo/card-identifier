docker build -t table-stream-classifier .
docker run -p 8080:8080 table-stream-classifier:latest

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin XXXXXXXXX.dkr.ecr.us-west-2.amazonaws.com
docker tag table-stream-classifier:latest XXXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/table-stream/mtg-classifier:latest
docker push XXXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/table-stream/mtg-classifier:latest
