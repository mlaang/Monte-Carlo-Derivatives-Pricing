#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>
//#include <fstream>
#include "CL\CL2.hpp"
using namespace std;

string file_to_string(char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f) {
		fseek(f, 0L, SEEK_END);
		unsigned int length = ftell(f);
		fseek(f, 0L, SEEK_SET);
		char* s = (char*)malloc(sizeof(char)*(length + 1));
		if (s)
			fread(s, 1, length, f);
		else return NULL;
		fclose(f);
		s[length] = '\0';
		return string(s);
	}
	else return NULL;
}

void handle_error(cl_int error_code, char* s) {
	if (CL_SUCCESS != error_code) {
		printf(s, error_code);
		exit(EXIT_FAILURE);
	}
}

void check_allocation(void* a, char* s) {
	if (!a) {
		cout << s;
		exit(EXIT_FAILURE);
	}
}


void handle_program_build_errors(cl_int error_code, cl_program program, cl_device_id device) {
	if (CL_SUCCESS != error_code) {
		char* build_log;
		size_t n_bytes;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n_bytes);
		build_log = (char*)malloc(sizeof(char)*n_bytes);
		if (build_log) {
			cl_int new_error_code = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*n_bytes, build_log, &n_bytes);
			printf("%s", build_log);
			free(build_log);
			handle_error(new_error_code, "clGetProgramBuildInfo(...) failed with error code %d.\n");
		}
		else printf("Out of memory.\n");
	}
}


int main() {
	cl_int error_code;

	string source_code = file_to_string("kernel.cl");
	cl::Program program(source_code, CL_TRUE, &error_code);
	handle_error(error_code, "Could not build program.\n cl::Program constructor failed with error %d.\n");

	cl::Device device = cl::Device::getDefault(&error_code);
	handle_error(error_code, "Could not get default device.\n cl::Device::getDefault(...) failed with error code %d.\n");

	cl::Context context = cl::Context::getDefault(&error_code);
	handle_error(error_code, "Could not get default context.\n cl::Context::getDefault(...) failed with error code %d.\n");

	cl::CommandQueue command_queue(context, device, cl::QueueProperties::None, &error_code);
	handle_error(error_code, "Could not create command queue. cl::DeviceCommandQueue constructor failed with error code %d.\n");

	cl::Kernel price_option(program, "price_option", &error_code);
	handle_error(error_code, "Could not construct speed test kernel.\n cl::Kernel constructor failed with error %d.\n");

	size_t workgroup_size = price_option.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &error_code);
	handle_error(error_code, "Could not get kernel workgroup size. cl::Kernel::getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(...) failed with error code %d.\n");

	cl_int N = 1000000000 / workgroup_size;
	cl_float S0 = 100,
		     T = 5.00f,
		     r = 0.05f,
		     sigma = 0.2f,
		     K = 70.0f;

	cl::size_type oversize_allocation = sizeof(cl_float) * (workgroup_size / 64 + 1) * 64;
	float* output = (cl_float*)_aligned_malloc(oversize_allocation, 4096);
	check_allocation(output, "Could not create output buffer.\n");

	cl::Buffer output_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, oversize_allocation, (void*)&output, &error_code);
	handle_error(error_code, "Could not create buffer. cl::Buffer constructor failed with error code %d.\n");

	price_option.setArg(0, output_buffer);
	price_option.setArg(1, S0);
	price_option.setArg(2, T);
	price_option.setArg(3, r);
	price_option.setArg(4, sigma);
	price_option.setArg(5, K);
	price_option.setArg(6, N);

	command_queue.enqueueNDRangeKernel(price_option, cl::NullRange, cl::NDRange(workgroup_size), cl::NullRange);
	command_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, oversize_allocation, output);

	float average = 0.0f;
	for (int i = 0; i != workgroup_size; ++i)
		average += output[i];
	average /= (float)workgroup_size;

	cout << "Simulated theoretical price of option is " << average << ".";

	return EXIT_SUCCESS;
}