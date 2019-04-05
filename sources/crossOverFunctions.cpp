/*
* @Author: Vyn
* @Date:   2019-04-05 11:47:29
* @Last Modified by:   Vyn
* @Last Modified time: 2019-04-05 12:03:21
*/

/*
* @Author: Vyn
* @Date:   2019-03-03 14:49:12
* @Last Modified by:   Vyn
* @Last Modified time: 2019-04-05 11:29:48
*/

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "Network.h"
#include "Neuron.h"
#include "Population.h"
#include "types.h"
#include "utils.h"

namespace vyn::neuralnetwork {

	Person_t	*GenerateChild(Population *population, Person_t *person1, Person_t *person2)
	{
		Person_t					*child;
		std::vector<Connection *>	childConnections;
		std::vector<Connection *>	person1Connections;
		std::vector<Connection *>	person2Connections;

		person1Connections = person1->network->GetConnections();
		person2Connections = person2->network->GetConnections();
		child = population->GenerateNewPerson();
		childConnections = child->network->GetConnections();
		for (int i = 0; i < childConnections.size(); ++i)
		{
			//if (i < childConnections.size() / 2)
			if (rand() % 2 == 0)
				childConnections[i]->SetWeight(person1Connections[i]->GetWeight());
			else
				childConnections[i]->SetWeight(person2Connections[i]->GetWeight());
			if ((value_t)rand() / (value_t)RAND_MAX < population->GetMutationChance())
				childConnections[i]->SetWeight(childConnections[i]->GetInput()->GetParentLayer()->NewWeightValue());
		}
		return (child);
	}

	void		DefaultCrossOverFunction(Population *population)
	{
		int	populationSize = population->GetSize();
		for (int i = 0; i < populationSize; ++i)
		{
			for (int j = 0; j < populationSize; ++j)
			{
				Person_t	*child;

				if (i != j)
				{
					child = GenerateChild(population, population->GetPersons()[i], population->GetPersons()[j]);
					population->AddPerson(child);
				}
			}
		}
	}
}