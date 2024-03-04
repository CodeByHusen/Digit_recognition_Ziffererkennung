#pragma once
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>

using namespace std;

class Reader
{
public:

	Reader()=default;
	Reader(const char fName[])
	{

		char c;
		int index = 0;

		ifstream infile;
		/// ios::in allows input (read operations) from a stream.
		infile.open(fName, ios::binary | ios::in);

		/// Calculates the number of characters in the transferred file
		f_Size = (int)filesize(fName);

		/// resize() resizes the vector 
		fMapV.resize(f_Size);

		/// The forloop continues until the last digit in the file is read
		for (int i = 0; i < f_Size; i++)
		{
			/// read()copies a block of data
            /// read(where items ares tored, number of items to read)
			infile.read(&c, 1);

			/// store the read letter in the Vector
			fMapV[i] = c;
		}
		/// Close the file, to create the file
		infile.close();
		read_Header();

	}

	//==============================================================
	/**
	 * @brief gives us the pixels of the desired image 
	*/
	float getPixel(int numIndex) const///numIndex of this picture stands for the number at which the next picture in the fMapV
	{
		/// numIndex stands for the gap between one picture and the next
        /// file train-images, this gap is 16 digits
		if (numIndex + data_Index >= fMapV.size())
		{
			return 0.0f;
		}
		/// store in a 16 bit hex number
		int a = fMapV[numIndex + data_Index];
		/// divide the number by 255 to return a number between 0 and 1 if it is not 0
		return ((float)a / 255);
	}

	int getLabel(int numIndex) const
	{
		/// 8 for label file this is the gap after which the first pixel of an image begins a digit
        /// numIndex the position of a pixel
		return fMapV[numIndex + data_Index];
	}

	//==============================================================
	/**
	 * @brief return the value of number of elements
	*/
	int get_N_Elements() const
	{
		return n_Elements;
	}

private:
	/// uses absolute positions in the stream
	std::ifstream::pos_type filesize(const char* filename)
	{
		/// ios::ate the stream's position indicator to the end when opening
        /// binary it is a binary file
		std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);

		/// tellg/() to know where the get pointer is
		return in.tellg();
	}

	

	//==============================================================

	/**
     * @brief read the header information from the file 
     * */
	void read_Header()
	{
		/// from the file at the position 0
		magic_Number = get_I32(0);

		if ((magic_Number != 2051) && (magic_Number != 2049))
		{
			throw std::runtime_error("Number isn't magic");
			return;
		}

		n_Elements = get_I32(4);

		if (magic_Number == 2051)
		{
			rows = get_I32(8);
			columns = get_I32(12);
			data_Index = 16;
		}
		else
		{
			data_Index = 8;
		}
	}

	//==============================================================

	/**
     * @brief use to retrieve a 32- bit interger from th file.
     * It takes an index as a parameter and uses bitwise operation to comnine the
     * values at theat index and the 3 follwing indices in an array"fMapV" to
     * return a single 32-bit integer
    */

	int32_t get_I32(int numIndex) const
	{
		return fMapV[numIndex + 3] | (fMapV[numIndex + 2] << 8) | (fMapV[numIndex + 1] << 16) | (fMapV[numIndex + 0] << 24);
	}

	//==============================================================

	int f_Size{};

	int32_t n_Elements;

	int32_t rows;

	int32_t columns;

	int32_t magic_Number;

	int32_t data_Index;

	std::vector<unsigned char> fMapV;
};
