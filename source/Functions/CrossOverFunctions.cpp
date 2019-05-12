/*
* @Author: Vyn
* @Date:   2019-04-05 11:47:29
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-09 15:51:49
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
#include "Types.h"
#include "Utils.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Person		*GenerateChild(Population *population, Person *person1, Person *person2)
		{
			Person					*child;

			child = population->GenerateNewPerson();
			const Connections &childConnections = child->GetNetwork()->GetConnections();
			const Connections &person1Connections = person1->GetNetwork()->GetConnections();
			const Connections &person2Connections = person2->GetNetwork()->GetConnections();
			for (int i = 0; i < childConnections.size(); ++i)
			{
				//if (i < childConnections.size() / 2)
				if (rand() % 2 == 0)
					childConnections[i]->SetWeight(person1Connections[i]->GetWeight());
				else
					childConnections[i]->SetWeight(person2Connections[i]->GetWeight());
				if ((Value)rand() / (Value)RAND_MAX < population->GetMutationChance())
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
					Person	*child;

					if (i != j)
					{
						child = GenerateChild(population, population->GetPersons()[i], population->GetPersons()[j]);
						population->AddPerson(child);
					}
				}
			}
		}
	}
}