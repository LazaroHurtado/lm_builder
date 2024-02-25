# DO NOT CHANGE this to just lm_builder, it will create a venv
# in the projects main folder and when a user run `make destory`
# it will remove every file in there
VENV_NAME = "lm_builder_venv"

setup: create_venv
	@echo "Activate the venv with: \`source ./$(VENV_NAME)/bin/activate\`"

create_venv:
	python3 -m venv ./$(VENV_NAME) ;\
	source ./$(VENV_NAME)/bin/activate ;\
	pip3 install -q -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

destroy: clean
	rm -rf ./$(VENV_NAME)