##
## EPITECH PROJECT, 2026
## MyTorch
## File description:
## Makefile
##

ANALYSER = my_torch_analyzer
GENERATOR = my_torch_generator
BONUS = my_torch_bonus

All: $(ANALYSER) $(GENERATOR) $(BONUS)

$(ANALYSER):
	ln src/$(ANALYSER)/entry_point $(ANALYSER)

$(GENERATOR):
	ln src/$(GENERATOR)/entry_point $(GENERATOR)

$(BONUS):
	ln src/bonus/entry_point $(BONUS)

clean:

fclean: clean
	$(RM) $(ANALYSER) $(GENERATOR) $(BONUS)

re: fclean all

.PHONY: all fclean clean re
