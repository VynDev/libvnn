/*
* @Author: Vyn
* @Date:   2019-04-02 16:44:28
* @Last Modified by:   Vyn
* @Last Modified time: 2019-04-05 11:51:45
*/

#include "Population.h"
#include "Neuron.h"
#include "Network.h"

namespace vyn::neuralnetwork {

	Population::Population(Network *(*f)() , int numberOfPerson)
	{
		networkCreationFunction = f;
		for (int i = 0; i < numberOfPerson; ++i)
		{
			AddPerson(GenerateNewPerson());
		}
	}

	void		Population::SetCrossOverFunction(int functionId)
	{
		if (functionId == CROSSOVER_FUNCTION_DEFAULT)
			SetCrossOverFunction(&DefaultCrossOverFunction);
	}

	void		Population::AddPerson(Person_t *person)
	{
		population.push_back(person);
	}

	Person_t	*Population::GenerateNewPerson()
	{
		Person_t *person;

		person = new Person_t;
		person->network = (*networkCreationFunction)();
		person->score = 0;
		person->id = nextId;
		person->population = this;
		++nextId;
		return (person);
	}	

	void		Population::CrossOver()
	{
		DeleteUnselectedPersons();
		(*crossOverFunction)(this);
		++currentGenerationId;
	}

	void		Population::DeleteUnselectedPersons()
	{
		for (int i = 0; i < population.size(); ++i)
		{
			if (population[i]->isSelected == false)
			{
				delete population[i]->network;
				delete population[i];
				population.erase(population.begin() + i);
				--i;
			}
			else
				population[i]->isSelected = false;
		}
	}

}