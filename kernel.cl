/* Generates pseudo-random floating points numbers in [-0.5,0.5). */
float random_uniform(unsigned int state[static const 5]) {
    //The state contains the current state of a pseudo-random number generator
	//this operation below uses a variant called XORWOW
    unsigned int s,
	             t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	state[4] += 362437;

	//t + state[4] should now contain 32 bits of pseudorandomness.
	//It's not unusual to compute something like (float)(t + state[4])/)(float)INT_MAX), but this
	//requires division. Another way to create a random number is to use bit operations to turn something
	//directly into a float

	//A 32-bit floating-point number has a sign bit in the highest position. Sice we are to generate a positive number
	//this is to be zero. Then there are eight bits of exponent and 23 bits for the fractional part. We will fill
	//the fractional part with bits of randomness from t + state[4] by shifting it to create zeros in the nine highest
	//bits. We then add the appropriate exponent and sign bits using bitwise or with the hexadecimal representation of 1.0f.
	//This produces a number that has the exponent of 1.0f, but a bunch of decimals instead of, up to something like 1.999.
	//Finally 1.5 is subtracted so as to obtain a number in [-0.5,0.5).

	return as_float(0x3F800000 | t + state[4] >> 9) - 1.5f;
}


/* Box-Muller transform */
float2 random_normal(unsigned int state[static const 5]) {
    float x, y, s;
	do {
	    x = random_uniform(state) + 0.5f;
		y = random_uniform(state) + 0.5f;
	} while(x == 0.0f);
	s = sqrt(-2.0f*log(x));
	return (float2)(s*cos(2.0f*M_PI_F * y), s*sin(2.0f*M_PI_F * y));
}

float2 random_lognormal(unsigned int state[static const 5], float mu, float sigma) {
    return exp(mu + sigma*random_normal(state));
}

/* S0	    initial stock price
   T        time to maturity
   r        interest rate
   sigma    volatility
   K        strike price
   N        iterations per work item
 */
__kernel void price_option(__global float* output, float S0, float T, float r, float sigma, float K, int N) {
    int id = get_global_id(0);
	float2 expectation = 0.0f;
	unsigned int state[5];
	state[0] = id;
	for(int i = 0; i != N; ++i)
	    expectation += exp(-r*T)*max(S0 * random_lognormal(state, (r - 0.5f*sigma*sigma)*T, sigma*sqrt(T)) - K, 0);
	output[id] = dot(expectation, (float2)(0.5f, 0.5f))/(float)N;
}