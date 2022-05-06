
.PHONY: build bash

build:
	docker-compose build --compress --parallel

run-db:
	docker-compose run --name db --rm db
run-admin:
	docker-compose run --name dbadmin --rm dbadmin


run-cmaes:
	docker-compose run --name "lcmaes" --rm lcmaes

bash:
	docker-compose run --name "lcmaes-bash" --rm lcmaes --entrypoint /bin/bash

redis:
	docker-compose up --name "redis" -d



build-without-cache:
	docker-compose build --no-cache --compress --parallel


#bash:
#	docker run --name lcmaesservice --rm -i -t docker_lcmaes bash
	#  -i interacive
	#  -t allocate tty
	#  --rm remove container after exit

complete_clean:
	#-@docker volume rm docker_python_virtual_environment
	docker image list | awk '(NR>1){print $3}' | xargs -n 1 docker rmi -f 


