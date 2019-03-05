/*
* @Author: Vyn
* @Date:   2019-02-01 15:33:35
* @Last Modified by:   Vyn
* @Last Modified time: 2019-02-15 13:24:13
*/

#include "Connection.h"

Connection::Connection()
{
	++nbConnection;

	connections.push_back(this);
}

int							Connection::nbConnection = 0;
std::vector<Connection *>	Connection::connections;