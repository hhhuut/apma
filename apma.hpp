#pragma once

#include <stdint.h>
#include <vector>

#ifdef _MSVC_LANG
#include <intrin.h>
#endif

#ifdef _DEBUG
#include <assert.h>
#endif

#include "radix_spline.h"

// Adaptive packed-memory array
template <class KeyType, class ValueType>
class pma
{
private:
	// Underlying structure
	struct _pma_storage
	{
		bool is_used;
		KeyType key{};
		ValueType value{};
		// TODO: Add stuff for concurrency

		_pma_storage() : is_used(false) {}
		_pma_storage(const _pma_storage& rhs) = delete; // Copying would mean allowing duplicate keys
		_pma_storage(_pma_storage&& rhs) noexcept { *this = std::move(rhs); }

		_pma_storage& operator=(const _pma_storage& rhs) = delete; // Copying would mean allowing duplicate keys
		_pma_storage& operator=(_pma_storage&& rhs) noexcept
		{
			// Copy elements
			is_used = rhs.is_used;
			key = rhs.key;
			value = rhs.value;

			// Leave rhs in a default state
			rhs.reset();

			return (*this);
		}

		void reset()
		{
			is_used = false;
			key = {};
			value = {};
		}
	};

public:
	// Iterator
	class _pma_const_iterator
	{
	private:
		using raw_const_iterator = typename std::vector<_pma_storage>::const_iterator;

	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = std::pair<KeyType, ValueType>;
		using difference_type = std::ptrdiff_t;
		using pointer = const value_type* const;
		using reference = const value_type&;

		_pma_const_iterator(raw_const_iterator _Piter, const pma* _Ppma) noexcept
			: _pma(_Ppma),
			_pmaIt(_Piter)
		{
			// Move given vector-iterator up to the next valid PMA element
			_advance_from_vector_begin();
			_construct_display_element();
		}

		reference operator*() const
		{
			return _val;
		}

		pointer operator->() const
		{
			return &_val;
		}

		// prefix (i.e. (++it)->a; )
		_pma_const_iterator& operator++()
		{
			if (_pmaIt < _pma->data_.end())
				++_pmaIt; // Move at least one vector element
			while (_pmaIt < _pma->data_.end() && !_pmaIt->is_used)
				++_pmaIt; // Go forward to next valid PMA element
			_construct_display_element();
			return (*this);
		}

		// postfix (i.e. (it++)->a; )
		_pma_const_iterator operator++(int)
		{
			_pma_const_iterator _tmp = *this;
			++(*this);
			return _tmp;
		}

		// prefix (i.e. (--it)->a; )
		_pma_const_iterator& operator--()
		{
			if (_pmaIt > _pma->data_.begin())
				--_pmaIt; // Move at least one vector element
			while (_pmaIt > _pma->data_.begin() && !_pmaIt->is_used)
				--_pmaIt; // Go backward to previous valid element
			_advance_from_vector_begin();
			_construct_display_element();
			return (*this);
		}

		// postfix (i.e. (it--)->a; )
		_pma_const_iterator operator--(int)
		{
			_pma_const_iterator _tmp = *this;
			--(*this);
			return _tmp;
		}

		bool operator==(const _pma_const_iterator& rhs)
		{
			return _pmaIt == rhs._pmaIt;
		}

		bool operator!=(const _pma_const_iterator& rhs)
		{
			return !(*this == rhs);
		}

	private:
		const pma* _pma; // PMA instance we're referencing
		raw_const_iterator _pmaIt; // PMA internal vector iterator (needed to advance this iterator)
		value_type _val; // Value to show to the outside (key-value-pair)

		void _advance_from_vector_begin()
		{
			// pma.end() == vector.end(), but pma.begin() points to the first valid PMA
			// element (so it might differ from vector.begin() ).
			if (_pmaIt == _pma->data_.begin())
			{
				while (_pmaIt < _pma->data_.end() && !_pmaIt->is_used)
					++_pmaIt;
			}
		}

		void _construct_display_element()
		{
			if (_pmaIt != _pma->data_.end())
				_val = value_type(_pmaIt->key, _pmaIt->value);
		}
	};

private:
	/* PMA constants */

	// Reserve 8 bits to allow for fixed point arithmetic.
	static constexpr uint64_t max_size = (1ULL << 56) - 1ULL;

	// Height-based (as opposed to depth-based) tree thresholds
	// Upper density thresholds
	static constexpr double up_h = 0.75; // root
	static constexpr double up_0 = 1.00; // leaves
	// Lower density thresholds
	static constexpr double low_h = 0.50; // root
	static constexpr double low_0 = 0.25; // leaves

	static constexpr uint8_t max_sparseness = 1 / low_0;
	static constexpr uint8_t largest_empty_segment = 1 * max_sparseness;

	/* General PMA fields */

	uint64_t num_elems = 0; // Number of elements
	uint64_t elem_capacity = (1ULL << largest_empty_segment); // Size of the underlying array
	uint32_t segment_size = largest_empty_segment; // Size of the segments
	uint64_t segment_count = elem_capacity / segment_size; // Number of segments
	uint32_t tree_height = (uint32_t)floor_log2(segment_count) + 1; // Height of the tree on top
	double delta_up = (up_0 - up_h) / tree_height; // Delta for upper density threshold
	double delta_low = (low_h - low_0) / tree_height; // Delta for lower density threshold
	std::vector<_pma_storage> data_; // Underlying storage

	bool is_indexed_ = true;
	rs::RadixSpline<KeyType> rs_;

public:
	using const_iterator = _pma_const_iterator;

	pma()
	{
		data_.resize(elem_capacity); // Initial size
		construct_spline();
	}
	pma(const pma& rhs) { *this = rhs; }
	pma(pma&& rhs) noexcept { *this = std::move(rhs); }
	// CONSTRUCTION FROM ARRAY IS UNTESTED!
	pma(const std::vector<std::pair<KeyType, ValueType>> vec)
	{
		// Construct PMA from regular array/vector
		size_t n = vec.size();

#ifdef _DEBUG
		assert(n > 0);
#endif
		num_elems = n;
		compute_capacity();
		tree_height = floor_log2(segment_count) + 1;
		delta_up = (up_0 - up_h) / tree_height;
		delta_low = (low_h - low_0) / tree_height;

		data_.resize(elem_capacity);

		for (size_t i = 0; i < elem_capacity; i++)
		{
			if (i < num_elems)
			{
				data_[i].key = vec[i].first;
				data_[i].value = vec[i].second;
				data_[i].is_used = true;
			}
		} // end of for
		spread(0, elem_capacity, num_elems);
	}

	pma& operator=(const pma& rhs)
	{
		num_elems = rhs.num_elems;
		elem_capacity = rhs.elem_capacity;
		segment_size = rhs.segment_size;
		segment_count = rhs.segment_count;
		tree_height = rhs.tree_height;
		delta_up = rhs.delta_up;
		delta_low = rhs.delta_low;
		data_ = rhs.data_;
		rs_ = rhs.rs_;
		return (*this);
	}
	pma& operator=(pma&& rhs) noexcept
	{
		// Copy all elements
		num_elems = rhs.num_elems;
		elem_capacity = rhs.elem_capacity;
		segment_size = rhs.segment_size;
		segment_count = rhs.segment_count;
		tree_height = rhs.tree_height;
		delta_up = rhs.delta_up;
		delta_low = rhs.delta_low;

		data_ = std::move(rhs.data_);
		rs_ = std::move(rhs.rs_);

		// Leave rhs in a default (but valid) state
		rhs.clear();

		return (*this);
	}

	size_t capacity() const { return elem_capacity; }
	size_t size() const { return num_elems; }

	const_iterator begin()
	{
		return const_iterator(data_.begin(), this);
	}

	const_iterator end()
	{
		return const_iterator(data_.end(), this);
	}

	void clear() noexcept
	{
		num_elems = 0;
		elem_capacity = (1ULL << largest_empty_segment);
		segment_size = largest_empty_segment;
		segment_count = elem_capacity / segment_size;
		tree_height = floor_log2(segment_count) + 1;
		double delta_up = (up_0 - up_h) / tree_height;
		double delta_low = (low_h - low_0) / tree_height;

		data_.clear();
		data_.resize(elem_capacity);

		construct_spline();
	}

	void construct_spline()
	{
		if (is_indexed_)
			return;

		if (num_elems == 0)
		{
			// Construct empty spline
			rs::Builder<KeyType> rsb(std::numeric_limits<KeyType>::min(), std::numeric_limits<KeyType>::max());
			rs_ = rsb.Finalize();
		}
		else
		{
			const auto min_key = begin()->first;
			const auto max_key = (--end())->first;
			auto prev_key = min_key - 1;

			// Dirty-hack so that the index of "min_key" is its actual index and not always the first vector index.
			rs::Builder<KeyType> rsb(prev_key, max_key);

			for (auto iter = data_.begin(); iter != data_.end(); iter++)
			{
				if (iter->is_used)
				{
					rsb.AddKey(iter->key);
					prev_key = iter->key;
				}
				else
					rsb.AddKey(prev_key); // "dummy" key so that the bounds are correct.
			} // end of for

			rs_ = rsb.Finalize();
		}

		is_indexed_ = true;
	}

	/// <summary>
	/// Perform a modified binary search, with $O(\log_2 n)$ comparisons, that
	/// allows gaps of size $O(1)$ in the array.
	/// </summary>
	/// <param name="key">The key to search for.</param>
	/// <param name="val">A reference to a variable that will be filled with the
	/// associated value. If the key isn't found, the content of val is undefined.</param>
	/// <returns>True if the value was found, otherwise false.</returns>
	bool find(KeyType key, ValueType& val) const
	{
		int64_t i;
		if (find_at(key, i))
		{
			val = data_[i].value;
			return true;
		}
		else
			return false;
	}

	bool radix_find_linear(KeyType key, ValueType& val)
	{
		construct_spline();

		auto searchbound = rs_.GetSearchBound(key);
		const auto end_it = data_.begin() + searchbound.end;

		auto iter = std::find_if(data_.begin() + searchbound.begin,
			end_it,
			[key](const _pma_storage& _left) { return _left.is_used && _left.key == key; });

		bool found = iter != end_it && iter->is_used && iter->key == key;

		if (found)
			val = iter->value;
		return found;
	}

	bool radix_find_binary(KeyType key, ValueType& val)
	{
		construct_spline();

		auto searchbound = rs_.GetSearchBound(key);

		int64_t i;
		// searchbound ist [from, to) but find expects [from, to]!
		if (find_between(key, searchbound.begin, searchbound.end - 1, i))
		{
			val = data_[i].value;
			return true;
		}
		else
			return false;
	}

	bool radix_find_exponential(KeyType key, ValueType& val)
	{
		construct_spline();

		auto searchbound = rs_.GetSearchBound(key);

		const auto start_ = searchbound.begin;
		const auto end_ = searchbound.end;

		if ((end_ - start_) <= 0)
			return false;

		size_t bound = 1; // as 2^0 = 1

		while (bound < (end_ - start_) && data_[start_ + bound].is_used && data_[start_ + bound].key < key)
			bound *= 2; // bound will increase as power of 2

		const auto offset = start_ + bound / 2;

		int64_t i;
		// searchbound is [from, to) but find expects [from, to]!
		if (find_between(key, offset, end_ - 1, i))
		{
			val = data_[i].value;
			return true;
		}
		else
			return false;
	}

	/// <summary>
	/// Remove the element with the given key.
	/// </summary>
	/// <param name="key">The key to remove.</param>
	/// <returns>True if the key was found and removed, otherwise false.</returns>
	bool remove(KeyType key)
	{
		int64_t i;
		if (find_at(key, i))
		{
			delete_at(i);
			is_indexed_ = false;
			return true;
		}
		else
			return false;
	}

	/// <summary>
	/// Insert a new element with the given key-value combination.
	/// </summary>
	/// <param name="key">The new element's key.</param>
	/// <param name="val">The new element's value.</param>
	/// <returns>True if the operation was successful, otherwise false.</returns>
	bool insert(KeyType key, ValueType val)
	{
		int64_t i;
		if (!find_at(key, i))
		{
			insert_after(i, key, val);
			is_indexed_ = false;
			return true;
		}
		else
			return false; // We do not allow duplicates
	}

private:
	/* Utility functions */

	// Returns the 1-based index of the last (most significant) bit set in x.
	inline uint64_t last_bit_set(uint64_t x) const
	{
#ifdef _DEBUG
		assert(x > 0);
#endif
#ifdef _MSVC_LANG
		return (sizeof(uint64_t) * 8 - __lzcnt64(x)); // MSVC (Windows)
#else
		return (sizeof(uint64_t) * 8 - __builtin_clzll(x)); // Linux
#endif
	}

	inline uint64_t floor_log2(uint64_t x) const {
		return (last_bit_set(x) - 1);
		// i.e. floor_log2(13) = 3, floor_log2(27) = 4, etc.
	}

	inline uint64_t ceil_log2(uint64_t x) const {
		return (last_bit_set(x - 1));
		// i.e. ceil_log2(13) = 4, ceil_log2(27) = 5, etc.
	}

	inline uint64_t ceil_div(uint64_t x, uint64_t y)
	{
#ifdef _DEBUG
		assert(x > 0);
#endif
		return (1 + ((x - 1) / y));
	}

	// Returns the largest power of 2 not greater than x ($2^{\lfloor \lg x \rfloor}$).
	inline uint64_t hyperfloor(uint64_t x)
	{
		return (1ULL << floor_log2(x));
	}

	// Returns the smallest power of 2 not less than x ($2^{\lceil \lg x \rceil}$).
	inline uint64_t hyperceil(uint64_t x)
	{
		return (1ULL << ceil_log2(x));
	}

	/* Internal functions */

	void compute_capacity()
	{
		segment_size = (uint32_t)ceil_log2(num_elems); // Ideal segment size
		segment_count = ceil_div(num_elems, segment_size); // Ideal number of segments

		// The number of segments has to be a power of 2, though.
		segment_count = hyperceil(segment_count);

		// Update the segment size accordingly
		segment_size = (uint32_t)ceil_div(num_elems, segment_count);
		elem_capacity = segment_size * segment_count;

		// Scale up as much as possible
		elem_capacity *= max_sparseness;
		segment_size *= max_sparseness;

#ifdef _DEBUG
		assert(elem_capacity <= max_size);
		assert(elem_capacity > num_elems);
#endif
	}

	void pack(size_t from, size_t to, uint64_t n)
	{
		// [from, to)
#ifdef _DEBUG
		assert(from < to);
#endif
		auto read_index = from;
		auto write_index = from;

		while (read_index < to)
		{
			if (data_[read_index].is_used)
			{
				if (read_index > write_index)
					data_[write_index] = std::move(data_[read_index]);
				++write_index;
			}
			++read_index;
		} // end of while

#ifdef _DEBUG
		assert(n == write_index - from);
#endif
	}

	void spread(size_t from, size_t to, uint64_t n)
	{
		// [from, to)
#ifdef _DEBUG
		assert(from < to);
#endif
		uint64_t capacity = to - from;
		uint64_t frequency = (capacity << 8) / n; // 8-bit fixed point arithmetic
		auto read_index = from + n - 1;
		auto write_index = (to << 8) - frequency;

		while ((write_index >> 8) > read_index)
		{
			data_[write_index >> 8] = std::move(data_[read_index]);

			--read_index;
			write_index -= frequency;
		} // end of while
	}

	void resize()
	{
		pack(0, elem_capacity, num_elems);
		compute_capacity();
		tree_height = (uint32_t)floor_log2(segment_count) + 1;
		delta_up = (up_0 - up_h) / tree_height;
		delta_low = (low_h - low_0) / tree_height;
		data_.resize(elem_capacity);
		for (auto i = num_elems; i < elem_capacity; i++)
			data_[i].reset(); // TODO: necessary?
		spread(0, elem_capacity, num_elems);
	}

	void rebalance(int64_t index)
	{
		// We're using signed indices here now since we need to perform
		// relative indexing and the range checking is much easier and
		// clearer with signed integral types.
		int64_t window_start, window_end;
		uint32_t height = 0;
		uint64_t occupancy = data_[index].is_used ? 1 : 0;
		int64_t left_index = index - 1;
		int64_t right_index = index + 1;
		double density, up_height, low_height;

		do
		{
			uint64_t window_size = segment_size * (1ULL << height);
			uint64_t window = index / window_size;
			window_start = window * window_size;
			window_end = window_start + window_size;

			while (left_index >= window_start)
			{
				if (data_[left_index].is_used)
					++occupancy;
				--left_index;
			} // end of while

			while (right_index < window_end)
			{
				if (data_[right_index].is_used)
					++occupancy;
				++right_index;
			} // end of while

			density = (double)occupancy / (double)window_size;
			up_height = up_0 - (height * delta_up);
			low_height = low_0 + (height * delta_low);
			++height;
		} while ((density < low_height || density >= up_height) && height < tree_height);

		if (density >= low_height && density < up_height)
		{
			// Found a window within threshold
			pack(window_start, window_end, occupancy);
			spread(window_start, window_end, occupancy);
		}
		else
			// Rebalance not possible without increasing the underlying array size.
			resize();
	}

	/// <summary>
	/// Perform a modified binary search.
	/// If the element is found, index holds the position of the element with the given key.
	/// If the element is not found, index holds the position of the predecessor or -1 if
	/// no predecessor exists.
	/// </summary>
	/// <param name="key">The key to search for.</param>
	/// <param name="index">The index of the found element, the index of its predecessor or -1.</param>
	/// <returns>True if the key is present, otherwise false.</returns>
	bool find_at(KeyType key, int64_t& index) const
	{
		return find_between(key, 0, elem_capacity - 1, index);
	}

	bool find_between(KeyType key, int64_t from, int64_t to, int64_t& index) const
	{
		// We're using signed indices here now since we need to perform
		// relative indexing and the range checking is much easier and
		// clearer with signed integral types.
		while (from <= to)
		{
			int64_t mid = from + (to - from) / 2;
			int64_t i = mid;

			// Start scanning left until we find a non-empty slot or 
			// we reach past the beginning of the sub-array.
			while (i >= from && !data_[i].is_used)
				--i;

			if (i < from)
			{
				// Everything between [from, mid] is empty.
				from = mid + 1;
			}
			else
			{
				if (data_[i].key == key)
				{
					index = i;
					return true;
				}
				else if (data_[i].key < key)
					from = mid + 1;
				else // data_[i].key > key
					to = i - 1;
			}
		} // end of while

		// Couldn't find 'key'. 'to' should hold its predecessor (unless it's empty).
		index = to;
		while (index >= 0 && !data_[index].is_used)
			--index;

		return false;
	}

	void delete_at(int64_t i)
	{
#ifdef _DEBUG
		assert(i >= 0);
		assert(i < elem_capacity);
#endif
		data_[i].reset();
		rebalance(i);
	}

	void insert_after(int64_t i, KeyType key, ValueType val)
	{
#ifdef _DEBUG
		assert(i >= -1);
		assert(i < (int64_t)elem_capacity);
		assert(i >= 0 && data_[i].is_used || i >= -1);
#endif
		int64_t j = i + 1;

		// Find an empty slot to the right of i.
		// There should be one close by.
		while (j < (int64_t)elem_capacity && data_[j].is_used)
			++j;

		if (j < (int64_t)elem_capacity)
		{
			// Found a slot.
			while (j > i + 1)
			{
				// Push elements to make space for the new element.
				// TODO: Find better way than iteration and pushing all elements single-file.
				data_[j] = std::move(data_[j - 1]);
				--j;
			} // end of while

			data_[i + 1].key = key;
			data_[i + 1].value = val;
			data_[i + 1].is_used = true;

			++i; // Update i to point to the new element.
		}
		else
		{
			// No empty space to the right side. Try left
			j = i - 1;

			while (j >= 0 && data_[j].is_used)
				--j;

			if (j >= 0)
			{
				// Found a slot (to the left).
				while (j < i)
				{
					// Push elements to make space for the new element.
					// TODO: Find better way than iteration and pushing all elements single-file.
					data_[j] = std::move(data_[j + 1]);
					++j;
				} // end of while

				data_[i].key = key;
				data_[i].value = val;
				data_[i].is_used = true;
			}
		}

		++num_elems;
		rebalance(i);
	}
};
