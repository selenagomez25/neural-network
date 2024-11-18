#pragma once
#include "weight.cpp"

class weight_factory
{

private:
public:
	weight_factory( std::vector<std::vector<double>>& weights,
		const int& rows,
		const int& col );
};

