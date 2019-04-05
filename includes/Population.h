#ifndef POPULATION_H
#define POPULATION_H

#include "Network.h"
#include "crossOverFunctions.h"
#include "types.h"

namespace vyn::neuralnetwork {

	class Population;

	typedef struct	Person_s
	{
		Network		*network = nullptr;
		bool		isSelected = false;
		int			score = 0;
		int			generationId;
		int			id;
		Population	*population = nullptr;
	}				Person_t;

	class Population {

	private:

		std::vector<Person_t *>	population;

		int			nextId = 0;
		int			currentGenerationId = 1;
		value_t		mutationChance = 0.05;

		Network		*(*networkCreationFunction)();
		void		(*crossOverFunction)(Population *) = &DefaultCrossOverFunction;

	public:

		Population(Network *(*f)(), int numberOfPerson);

		std::vector<Person_t *>	&GetPersons() {return (population);};
		int						GetSize() const {return (population.size());};
		int						GetCurrentGenerationId() const {return (currentGenerationId);};

		void					SetMutationChance(value_t value) {mutationChance = value;};
		value_t					GetMutationChance() const {return (mutationChance);};

		void					AddPerson(Person_t *person);
		Person_t				*GenerateNewPerson();

		void					SetCrossOverFunction(int functionId);
		void					SetCrossOverFunction(void (*f)(Population *)) {crossOverFunction = f;};
		void					CrossOver();

		void					DeleteUnselectedPersons();
	};
}
#endif