##
## EPITECH PROJECT, 2026
## MyTorch
## File description:
## Makefile
##

ANALYSER = my_torch_analyzer
GENERATOR = my_torch_generator
BONUS = my_torch_bonus
HYPEROPTIMIZER = my_torch_hyperoptimizer

all: $(ANALYSER) $(GENERATOR) $(BONUS) $(HYPEROPTIMIZER)

$(ANALYSER):
	ln src/$(ANALYSER)/entry_point $(ANALYSER)

$(GENERATOR):
	ln src/$(GENERATOR)/entry_point $(GENERATOR)

$(BONUS):
	ln src/bonus/entry_point $(BONUS)

$(HYPEROPTIMIZER):
	ln src/$(HYPEROPTIMIZER)/entry_point $(HYPEROPTIMIZER)

clean:

fclean: clean
	$(RM) $(ANALYSER) $(GENERATOR) $(BONUS) $(HYPEROPTIMIZER)

re: fclean all

.PHONY: all fclean clean re
