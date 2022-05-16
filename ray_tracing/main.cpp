//
//  main.cpp
//  ray_tracing
//
//  Created by ianslayer on 2022/2/21.
//

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS //let msvc happy to compile stb_image_write.h
#include "stb_image_write.h"

#include <cstdlib>
#include <vector>
#include <numeric>
#include <assert.h>
#include <chrono>

#if defined(__APPLE__) && defined(__MACH__)
    #include "TargetConditionals.h"
    #if TARGET_OS_OSX
      // Put CPU-independent macOS code here.
      #if TARGET_CPU_ARM64
        #include "sse2neon.h"
      #elif TARGET_CPU_X86_64
        // Put 64-bit Intel macOS code here.
        #include <x86intrin.h>
      #endif
    #elif TARGET_OS_MACCATALYST
       // Put Mac Catalyst-specific code here.
    #elif TARGET_OS_IOS
      // Put iOS-specific code here.
    #endif
#elif defined (_WIN32) || defined(_WIN64)
    #include <immintrin.h>
#else
    #error "unsupported platform"
#endif

#define PI 3.1415926535f

union simd_value
{
    __m128 v;
    __m128i iv;
    uint32_t i[4];
    float f[4];
};

uint32_t wang_hash(uint32_t& seed)
{
    seed =  (seed ^ uint32_t(61)) ^ (seed >> uint32_t(16));
    seed *= uint32_t(9);
    seed = seed ^ (seed >> uint32_t(4));
    seed *= uint32_t(0x27d4eb2d);
    seed = seed ^ (seed >> uint32_t(15));
    
    return seed;
}

uint32_t pcg_hash(uint32_t& input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    input =(word >> 22u) ^ word;
    return input;
}

float random_float_0_1(uint32_t& seed)
{
    return float(wang_hash(seed)) / 4294967296.f;
}

union vec3_t
{
    struct {
        float x, y, z;
    };
    
    const vec3_t& operator -= (const vec3_t& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        
        return *this;
    }
    
    const vec3_t& operator += (const vec3_t& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        
        return *this;
    }
    
    const vec3_t& operator *= (float s)
    {
        x *= s;
        y *= s;
        z *= s;
        
        return *this;
    }
    
    const vec3_t& operator *= (const vec3_t& rhs)
    {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        
        return *this;
    }
    
    const vec3_t& operator /= (float s)
    {
        assert (s != 0);
        x /= s;
        y /= s;
        z /= s;
        
        return *this;
    }
    
    const vec3_t& operator /= (const vec3_t& rhs)
    {
        assert( rhs.x != 0 );
        assert( rhs.y != 0 );
        assert( rhs.z != 0 );
        
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        
        return *this;
    }
};

inline vec3_t operator - (vec3_t v0, const vec3_t& v1)
{
    v0 -= v1;
    return v0;
}

inline vec3_t operator + (vec3_t v0, const vec3_t& v1)
{
    v0 += v1;
    return v0;
}

inline vec3_t operator * (vec3_t v, float s)
{
    v *= s;
    return v;
}

inline vec3_t operator * (float s, vec3_t v)
{
    v *= s;
    return v;
}

inline vec3_t operator * (vec3_t v0, const vec3_t& v1)
{
    v0 *= v1;
    return v0;
}

inline vec3_t operator / (vec3_t v, float s)
{
    v /= s;
    return v;
}

inline vec3_t operator / (vec3_t v0, const vec3_t& v1)
{
    v0 /= v1;
    return v0;
}

float length(const vec3_t& v)
{
    float length = sqrtf( v.x * v.x + v.y * v.y + v.z * v.z);
    return length;
}

inline vec3_t normalize(vec3_t v)
{
    float length = sqrtf( v.x * v.x + v.y * v.y + v.z * v.z);
    assert(length != 0);
    v /= length;
    return v;
}

inline vec3_t cross(vec3_t v0, vec3_t v1)
{
    return { v0.y * v1.z - v1.y * v0.z, v0.z * v1.x - v1.z * v0.x, v0.x * v1.y - v1.x * v0.y };
}

inline float dot(vec3_t v0, vec3_t v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

struct vec3_soa_t
{
    __m128 x;
    __m128 y;
    __m128 z;
    
    const vec3_soa_t& operator -= (const vec3_soa_t& rhs)
    {
        x = _mm_sub_ps(x, rhs.x);
        y = _mm_sub_ps(y, rhs.y);
        z = _mm_sub_ps(z, rhs.z);
        
        return *this;
    }
    
    const vec3_soa_t& operator += (const vec3_soa_t& rhs)
    {
        x = _mm_add_ps(x, rhs.x);
        y = _mm_add_ps(y, rhs.y);
        z = _mm_add_ps(z, rhs.z);
        
        return *this;
    }
    
    const vec3_soa_t& operator *= (__m128 s)
    {
        x = _mm_mul_ps(x, s);
        y = _mm_mul_ps(y, s);
        z = _mm_mul_ps(z, s);
        
        return *this;
    }
    
    const vec3_soa_t& operator *= (const vec3_soa_t& rhs)
    {
        x = _mm_mul_ps(x, rhs.x);
        y = _mm_mul_ps(y, rhs.y);
        z = _mm_mul_ps(z, rhs.z);
        
        return *this;
    }
    
    const vec3_soa_t& operator /= (__m128 s)
    {
        x = _mm_div_ps(x, s);
        y = _mm_div_ps(y, s);
        z = _mm_div_ps(z, s);
        
        return *this;
    }
    
    const vec3_soa_t& operator /= (const vec3_soa_t& rhs)
    {
        x = _mm_div_ps(x, rhs.x);
        y = _mm_div_ps(y, rhs.y);
        z = _mm_div_ps(z, rhs.z);
        
        return *this;
    }
};

vec3_soa_t vec3_to_soa(const vec3_t& vec)
{
    vec3_soa_t result;
    
    result.x = _mm_set_ps1(vec.x);
    result.y = _mm_set_ps1(vec.y);
    result.z = _mm_set_ps1(vec.z);
    
    return result;
}

inline vec3_soa_t operator - (vec3_soa_t v0, const vec3_soa_t& v1)
{
    v0 -= v1;
    return v0;
}

inline vec3_soa_t operator + (vec3_soa_t v0, const vec3_soa_t& v1)
{
    v0 += v1;
    return v0;
}

inline vec3_soa_t operator * (vec3_soa_t v, __m128 s)
{
    v *= s;
    return v;
}

inline vec3_soa_t operator * (__m128 s, vec3_soa_t v)
{
    v *= s;
    return v;
}

inline vec3_soa_t operator * (vec3_soa_t v0, const vec3_soa_t& v1)
{
    v0 *= v1;
    return v0;
}

inline vec3_soa_t operator / (vec3_soa_t v, __m128 s)
{
    v /= s;
    return v;
}

inline vec3_soa_t operator / (vec3_soa_t v0, vec3_soa_t v1)
{
    v0 /= v1;
    return v0;
}

inline vec3_soa_t cross(const vec3_soa_t& v0, const vec3_soa_t& v1)
{
    vec3_soa_t result;
    
    result.x = _mm_sub_ps(_mm_mul_ps(v0.y, v1.z), _mm_mul_ps(v1.y, v0.z));
    result.y = _mm_sub_ps(_mm_mul_ps(v0.z, v1.x), _mm_mul_ps(v1.z, v0.x));
    result.z = _mm_sub_ps(_mm_mul_ps(v0.x, v1.y), _mm_mul_ps(v1.x, v0.y));
    
    return result;
}

inline __m128 dot(const vec3_soa_t& v0, const vec3_soa_t& v1)
{
    __m128 result;
    
    __m128 x = _mm_mul_ps(v0.x, v1.x);
    __m128 y = _mm_mul_ps(v0.y, v1.y);
    __m128 z = _mm_mul_ps(v0.z, v1.z);
    
    result = _mm_add_ps(x, y);
    result = _mm_add_ps(result, z);
    
    return result;
}

__m128 length(const vec3_soa_t& v)
{
    __m128 result = dot(v, v);
    result = _mm_sqrt_ps(result);
    
    return result;
}

vec3_t random_unit_vector(uint32_t& seed)
{
    float z = random_float_0_1(seed) * 2.0f - 1.0f;
    float a = random_float_0_1(seed) * 2.0f * PI;
    float r = sqrtf(1.f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    
    return vec3_t{x, y, z};
}


struct triangle_t
{
    vec3_t p0, p1, p2;
    uint32_t material_id;
};

vec3_t normal(const triangle_t& triangle)
{
    vec3_t v0 = triangle.p0 - triangle.p1;
    vec3_t v1 = triangle.p2 - triangle.p1;
    
    return normalize(cross(v1, v0));
}

struct material_t
{
    vec3_t albedo;
    vec3_t emissive;
};

#if defined (_WIN32) || defined (_WIN64)
void* aligned_alloc(size_t alignment, size_t size)
{
    return _aligned_malloc(size, alignment);
}
#endif

struct triangle_soa_t
{
    triangle_soa_t()
    {
        
    }
    
    void resize(size_t _simd_array_length)
    {
        simd_array_length = _simd_array_length;
        
        size_t allocation_size = simd_array_length * sizeof(__m128);
        
        p0_x = (__m128*) aligned_alloc(16, allocation_size);
        p0_y = (__m128*) aligned_alloc(16, allocation_size);
        p0_z = (__m128*) aligned_alloc(16, allocation_size);
        
        p1_x = (__m128*) aligned_alloc(16, allocation_size);
        p1_y = (__m128*) aligned_alloc(16, allocation_size);
        p1_z = (__m128*) aligned_alloc(16, allocation_size);
        
        p2_x = (__m128*) aligned_alloc(16, allocation_size);
        p2_y = (__m128*) aligned_alloc(16, allocation_size);
        p2_z = (__m128*) aligned_alloc(16, allocation_size);
        
        assert(size_t(p0_x) % 16 == 0 );
        assert(size_t(p0_y) % 16 == 0 );
        assert(size_t(p0_z) % 16 == 0 );

        assert(size_t(p1_x) % 16 == 0 );
        assert(size_t(p1_y) % 16 == 0 );
        assert(size_t(p1_z) % 16 == 0 );
        
        assert(size_t(p2_x) % 16 == 0 );
        assert(size_t(p2_y) % 16 == 0 );
        assert(size_t(p2_z) % 16 == 0 );

    }
    
    void release()
    {
        free(p0_x);
        free(p0_y);
        free(p0_z);
        
        free(p1_x);
        free(p1_y);
        free(p1_z);
        
        free(p2_x);
        free(p2_y);
        free(p2_z);
    }
    
    ~triangle_soa_t()
    {
        release();
    }
    
    __m128* p0_x = 0;
    __m128* p0_y = 0;
    __m128* p0_z = 0;
    
    __m128* p1_x = 0;
    __m128* p1_y = 0;
    __m128* p1_z = 0;
    
    __m128* p2_x = 0;
    __m128* p2_y = 0;
    __m128* p2_z = 0;
    
    size_t simd_array_length = 0;
    
};

struct scene_t
{
    std::vector<triangle_t> triangle_list;
    triangle_soa_t triangles;
    
    std::vector<material_t> material_list;
};

struct camera_t
{
    vec3_t look_at;
    vec3_t right;
    vec3_t down;
    
    vec3_t position;
    
    float fov;
};

struct ray_t
{
    vec3_t origin;
    vec3_t direction;
};

struct ray_triangle_intersect_info_t
{
    bool intersect;
    float u, v, t;
};

struct ray_scene_intersect_info_t
{
    int triangle_index;
    
    ray_triangle_intersect_info_t triangle_intersect_info;
};

inline ray_triangle_intersect_info_t intersect(const ray_t& ray, const triangle_t& triangle)
{
    ray_triangle_intersect_info_t intersect_info = {false, 0.f, 0.f, 0.f};
    
    vec3_t e0 = triangle.p1 - triangle.p0;
    vec3_t e1 = triangle.p2 - triangle.p0;
    
    vec3_t q = cross(ray.direction, e1);
    float det = dot(e0, q);
    
    float epsilon = 1e-5f;
    
    if (det > -epsilon && det < epsilon)
        return intersect_info;
    
    float f = 1.f / det;
    
    vec3_t s = ray.origin - triangle.p0;
    float u = f * dot(s, q);
    
    if(u < 0.f)
        return intersect_info;
    
    vec3_t r = cross(s, e0);
    float v = f * dot(ray.direction, r);
    
    if(v < 0.f || (u + v) > 1.f)
        return intersect_info;
    
    float t = f * dot(e1, r);
    
    intersect_info.intersect = true;
    intersect_info.u = u;
    intersect_info.v = v;
    intersect_info.t = t;
    
    return intersect_info;
}

inline ray_scene_intersect_info_t intersect(const ray_t& ray, const scene_t& scene)
{
    ray_scene_intersect_info_t scene_intersect_info = {-1};
    
    float hit_min = std::numeric_limits<float>::max() ;
    
    for(size_t triangle_index = 0; triangle_index < scene.triangle_list.size(); triangle_index++) {
        const triangle_t& triangle = scene.triangle_list[triangle_index];
        ray_triangle_intersect_info_t intersect_info = intersect(ray, triangle);
        
        if(intersect_info.intersect) {
            
            if(intersect_info.t < hit_min && intersect_info.t > 0) {
                
                scene_intersect_info.triangle_index = (int) triangle_index;
                scene_intersect_info.triangle_intersect_info = intersect_info;
                
                hit_min = intersect_info.t;

            }
        }
        
    }
    
    return scene_intersect_info;
}

struct ray_triangle_intersect_info_soa_t
{
    __m128i intersect;
    
    __m128 u;
    __m128 v;
    __m128 t;
};

inline ray_triangle_intersect_info_t extract(const ray_triangle_intersect_info_soa_t& info, int index)
{
    ray_triangle_intersect_info_t result;
    
    simd_value intersect;
    simd_value u;
    simd_value v;
    simd_value t;
    
    intersect.iv = info.intersect;
    u.v = info.u;
    v.v = info.v;
    t.v = info.t;
    
    result.intersect = (bool) intersect.i[index];
    result.u = u.f[index];
    result.v = v.f[index];
    result.t = t.f[index];
    
    return result;
}

inline  ray_triangle_intersect_info_soa_t intersect_simd(const ray_t& ray, const triangle_soa_t& triangles, size_t simd_index)
{
    ray_triangle_intersect_info_soa_t intersect_info = {};
    
    vec3_soa_t p0 = { triangles.p0_x[simd_index], triangles.p0_y[simd_index], triangles.p0_z[simd_index]};
    vec3_soa_t p1 = { triangles.p1_x[simd_index], triangles.p1_y[simd_index], triangles.p1_z[simd_index]};
    vec3_soa_t p2 = { triangles.p2_x[simd_index], triangles.p2_y[simd_index], triangles.p2_z[simd_index]};
    
    vec3_soa_t e0 = p1 - p0;
    vec3_soa_t e1 = p2 - p0;
    
    vec3_soa_t ray_dir = vec3_to_soa(ray.direction);
    
    vec3_soa_t q = cross(ray_dir, e1);
    __m128 det = dot(e0, q);
    
    __m128i mask = _mm_set_epi32(-1, -1, -1, -1);
    
    __m128 c_epsilon = _mm_set_ps1(1e-5f);
    __m128 c_m_epsilon = _mm_set_ps1(-1e-5f);
    
    __m128i det_gt_epsilon = _mm_castps_si128(_mm_cmple_ps(det, c_m_epsilon));
    __m128i det_lt_epsilon = _mm_castps_si128(_mm_cmpge_ps(det, c_epsilon));
    
    __m128i test_det = _mm_or_si128(det_gt_epsilon, det_lt_epsilon);
    
    mask = _mm_and_si128(mask, test_det);
    
    //if (det > -epsilon && det < epsilon)
    //    return intersect_info;
    
    int mask_all_zero = _mm_test_all_zeros(mask, mask);
    
    if(mask_all_zero) {
        return intersect_info;
    }
    
    __m128 f = _mm_div_ps(_mm_set_ps1(1.f), det);
    
    vec3_soa_t ray_origin = vec3_to_soa(ray.origin);
    vec3_soa_t s = ray_origin - p0;
    __m128 u = _mm_mul_ps(f, dot(s, q));
    
    __m128 c0 = _mm_set_ps1(0.f);
    
    __m128 test_u = _mm_cmpge_ps(u, c0);
    
    //if(u < 0.f)
    //    return intersect_info;
    
    mask = _mm_and_si128(mask, _mm_castps_si128(test_u));
    mask_all_zero = _mm_test_all_zeros(mask, mask);
    
    if(mask_all_zero) {
        return intersect_info;
    }
    
    vec3_soa_t r = cross(s, e0);
    __m128 v = _mm_mul_ps(f, dot(ray_dir, r));
    
    __m128i test_v = _mm_castps_si128(_mm_cmpge_ps(v, c0));
    __m128 add_uv = _mm_add_ps(u, v);

    __m128 c1 = _mm_set_ps1(1.f);
    
    __m128i test_uv = _mm_castps_si128(_mm_cmple_ps(add_uv, c1));
    
    mask = _mm_and_si128(mask, test_v);
    mask = _mm_and_si128(mask, test_uv);
    mask_all_zero = _mm_test_all_zeros(mask, mask);
    
    if(mask_all_zero) {
        return intersect_info;
    }
    //if(v < 0.f || (u + v) > 1.f)
    //    return intersect_info;
    
    __m128 t = _mm_mul_ps(f, dot(e1, r));
    
    intersect_info.intersect = mask;
    intersect_info.u = u;
    intersect_info.v = v;
    intersect_info.t = t;
    
    return intersect_info;
}

inline ray_scene_intersect_info_t intersect_simd(const ray_t& ray, const scene_t& scene)
{
    ray_scene_intersect_info_t scene_intersect_info = {-1};
    
    float hit_min = std::numeric_limits<float>::max();
    
    ray_triangle_intersect_info_soa_t intersect_info = {};
    int intersect_mask = 0;
    
    int triangles_simd_length = (int)scene.triangles.simd_array_length;
    for(int simd_index = 0; simd_index < triangles_simd_length; simd_index++) {
        
        intersect_info = intersect_simd(ray, scene.triangles, simd_index);
        
        intersect_mask = _mm_movemask_ps( _mm_castsi128_ps( intersect_info.intersect));
        
        if(intersect_mask) {
            int triangle_base_index = 4 * simd_index;
            
            simd_value t;
            t.v = intersect_info.t;
            
            for(int i = 0; i < 4; i++) {
                
                if(intersect_mask & (1 << i)) {
                    
                    if(t.f[i] < hit_min && t.f[i] > 0) {
                        scene_intersect_info.triangle_index = triangle_base_index + i;
                        scene_intersect_info.triangle_intersect_info = extract(intersect_info, i);
                        
                        hit_min = t.f[i];
                        
                    }
                }
            }
        }
        
    }
    
    return scene_intersect_info;
}

inline vec3_t flip_yz(vec3_t p)
{
    return vec3_t{p.x, p.z, p.y};
}

inline void push_face(scene_t* scene, vec3_t p0, vec3_t p1, vec3_t p2, vec3_t p3, uint32_t color_index)
{
    p0 = flip_yz(p0);
    p1 = flip_yz(p1);
    p2 = flip_yz(p2);
    p3 = flip_yz(p3);
    
    triangle_t triangle[2];
    
    triangle[0].p0 = p1;
    triangle[0].p1 = p0;
    triangle[0].p2 = p2;
    triangle[0].material_id = color_index;
    
    triangle[1].p0 = p2;
    triangle[1].p1 = p0;
    triangle[1].p2 = p3;
    triangle[1].material_id = color_index;
    
    scene->triangle_list.push_back(triangle[0]);
    scene->triangle_list.push_back(triangle[1]);
    
}

inline uint32_t push_material(scene_t* scene, vec3_t color, vec3_t emissive)
{
    material_t material = {color, emissive};
    
    scene->material_list.push_back(material);
    return (uint32_t) scene->material_list.size() - 1;
}

void transpose_scene_soa(scene_t* scene)
{
    size_t triangles_align4 = scene->triangle_list.size() & (size_t) (~3);
    
    size_t simd_array_length = (scene->triangle_list.size() + 3) / 4;
    
    scene->triangles.resize(simd_array_length);
    
    size_t simd_index = 0;
    for(size_t triangle_index = 0; triangle_index < triangles_align4; triangle_index += 4){
        const triangle_t* triangles = &scene->triangle_list[triangle_index];
        
        __m128 p0x = _mm_set_ps(triangles[3].p0.x, triangles[2].p0.x, triangles[1].p0.x, triangles[0].p0.x);
        __m128 p0y = _mm_set_ps(triangles[3].p0.y, triangles[2].p0.y, triangles[1].p0.y, triangles[0].p0.y);
        __m128 p0z = _mm_set_ps(triangles[3].p0.z, triangles[2].p0.z, triangles[1].p0.z, triangles[0].p0.z);

        scene->triangles.p0_x[simd_index] = p0x;
        scene->triangles.p0_y[simd_index] = p0y;
        scene->triangles.p0_z[simd_index] = p0z;
        
        __m128 p1x = _mm_set_ps(triangles[3].p1.x, triangles[2].p1.x, triangles[1].p1.x, triangles[0].p1.x);
        __m128 p1y = _mm_set_ps(triangles[3].p1.y, triangles[2].p1.y, triangles[1].p1.y, triangles[0].p1.y);
        __m128 p1z = _mm_set_ps(triangles[3].p1.z, triangles[2].p1.z, triangles[1].p1.z, triangles[0].p1.z);
    
        scene->triangles.p1_x[simd_index] = p1x;
        scene->triangles.p1_y[simd_index] = p1y;
        scene->triangles.p1_z[simd_index] = p1z;
        
        __m128 p2x = _mm_set_ps(triangles[3].p2.x, triangles[2].p2.x, triangles[1].p2.x, triangles[0].p2.x);
        __m128 p2y = _mm_set_ps(triangles[3].p2.y, triangles[2].p2.y, triangles[1].p2.y, triangles[0].p2.y);
        __m128 p2z = _mm_set_ps(triangles[3].p2.z, triangles[2].p2.z, triangles[1].p2.z, triangles[0].p2.z);
        
        scene->triangles.p2_x[simd_index] = p2x;
        scene->triangles.p2_y[simd_index] = p2y;
        scene->triangles.p2_z[simd_index] = p2z;
        
        simd_index++;
        
    }
    
    triangle_t triangles[4] = {0};
    
    //left triangles
    {
        for(size_t triangle_index = triangles_align4, i = 0; triangle_index < scene->triangle_list.size(); triangle_index++, i++) {
            triangles[i] = scene->triangle_list[triangle_index];
        }
        
        __m128 p0x = _mm_set_ps(triangles[3].p0.x, triangles[2].p0.x, triangles[1].p0.x, triangles[0].p0.x);
        __m128 p0y = _mm_set_ps(triangles[3].p0.y, triangles[2].p0.y, triangles[1].p0.y, triangles[0].p0.y);
        __m128 p0z = _mm_set_ps(triangles[3].p0.z, triangles[2].p0.z, triangles[1].p0.z, triangles[0].p0.z);

        scene->triangles.p0_x[simd_index] = p0x;
        scene->triangles.p0_y[simd_index] = p0y;
        scene->triangles.p0_z[simd_index] = p0z;
        
        __m128 p1x = _mm_set_ps(triangles[3].p1.x, triangles[2].p1.x, triangles[1].p1.x, triangles[0].p1.x);
        __m128 p1y = _mm_set_ps(triangles[3].p1.y, triangles[2].p1.y, triangles[1].p1.y, triangles[0].p1.y);
        __m128 p1z = _mm_set_ps(triangles[3].p1.z, triangles[2].p1.z, triangles[1].p1.z, triangles[0].p1.z);
    
        scene->triangles.p1_x[simd_index] = p1x;
        scene->triangles.p1_y[simd_index] = p1y;
        scene->triangles.p1_z[simd_index] = p1z;
        
        __m128 p2x = _mm_set_ps(triangles[3].p2.x, triangles[2].p2.x, triangles[1].p2.x, triangles[0].p2.x);
        __m128 p2y = _mm_set_ps(triangles[3].p2.y, triangles[2].p2.y, triangles[1].p2.y, triangles[0].p2.y);
        __m128 p2z = _mm_set_ps(triangles[3].p2.z, triangles[2].p2.z, triangles[1].p2.z, triangles[0].p2.z);
        
        scene->triangles.p2_x[simd_index] = p2x;
        scene->triangles.p2_y[simd_index] = p2y;
        scene->triangles.p2_z[simd_index] = p2z;
        
        simd_index++;
    }
    
}

void construct_cornell_box_scene(scene_t* scene)
{
    vec3_t white = {0.5, 0.5, 0.5};
    vec3_t green = {0.0, 0.5, 0.0};
    vec3_t red = {0.5, 0.0, 0.0};
    vec3_t black = {0, 0, 0};
    vec3_t light = {100.0, 100.0, 100.0};
    
    uint32_t white_index = push_material(scene, white, black);
    uint32_t green_index = push_material(scene, green, black);
    uint32_t red_index = push_material(scene, red, black);
    uint32_t black_index = push_material(scene, black, black);
    uint32_t light_index = push_material(scene, white, light);
    
    push_face(scene,
              vec3_t{552.8f, 0, 0},
              vec3_t{0, 0, 0},
              vec3_t{0, 0, 559.2f},
              vec3_t{549.6f, 0, 559.2f},
              white_index); //floor
    
    push_face(scene,
              vec3_t{130.f, 0.0, 65.f},
              vec3_t{82.f, 0, 225.f},
              vec3_t{240.f, 0, 272.f},
              vec3_t{290.f, 0, 114.f},
              black_index); //hole 1
    
    push_face(scene,
              vec3_t{423.f, 0, 247.f},
              vec3_t{265.f, 0, 296.f},
              vec3_t{314.f, 0, 456.f},
              vec3_t{472.f, 0, 406.f},
              black_index); //hole 2
    
    push_face(scene,
              vec3_t{343.f, 548.7f, 227.f}, //original y is 548.8, move down a bit to remove z fighting
              vec3_t{343.f, 548.7f, 332.f},
              vec3_t{213.f, 548.7f, 332.f},
              vec3_t{213.f, 548.7f, 227.f},
              light_index); //light
    
    push_face(scene,
              vec3_t{556.f, 548.8f, 0},
              vec3_t{556.f, 548.8f, 559.2f},
              vec3_t{0, 548.8f, 559.2f},
              vec3_t{0, 548.8f, 0},
              white_index); //ceiling
    
    push_face(scene,
              vec3_t{343.f, 548.8f, 227.f},
              vec3_t{343.f, 548.8f, 332.f},
              vec3_t{213.f, 548.8f, 332.f},
              vec3_t{213.f, 548.8f, 227.f},
              black_index); //ceiling hole
    
    push_face(scene,
              vec3_t{549.6f, 0, 559.2f},
              vec3_t{0, 0, 559.2f},
              vec3_t{0, 548.8f, 559.2f},
              vec3_t{556.f, 548.8f, 559.2f},
              white_index); //back wall
    
    push_face(scene,
              vec3_t{0, 0, 559.2f},
              vec3_t{0, 0, 0},
              vec3_t{0, 548.8f, 0},
              vec3_t{0, 548.8f, 559.2f},
              green_index); //right wall
    
    push_face(scene,
              vec3_t{552.8f, 0, 0},
              vec3_t{549.6f, 0, 559.2f},
              vec3_t{556.f, 548.8f, 559.2f},
              vec3_t{556.f, 548.8f, 0},
              red_index); //left wall
    
    //short block
    push_face(scene,
              vec3_t{130.f, 165.f, 65.f},
              vec3_t{82.f, 165.f, 225.f},
              vec3_t{240.f, 165.f, 272.f},
              vec3_t{290.f, 165.f, 114.f},
              white_index);
    
    push_face(scene,
              vec3_t{290.f, 0, 114.f},
              vec3_t{290.f, 165.f, 114.f},
              vec3_t{240.f, 165.f, 272.f},
              vec3_t{240.f, 0, 272.f},
              white_index);
    
    push_face(scene,
              vec3_t{130.f, 0, 65.f},
              vec3_t{130.f, 165.f, 65.f},
              vec3_t{290.f, 165.f, 114.f},
              vec3_t{290.f, 0, 114.f},
              white_index);
    
    push_face(scene,
              vec3_t{82.f, 0, 225.f},
              vec3_t{82.f, 165.f, 225.f},
              vec3_t{130.f, 165.f, 65.f},
              vec3_t{130.f, 0, 65.f},
              white_index);
    
    push_face(scene,
              vec3_t{240.f, 0, 272.f},
              vec3_t{240.f, 165.f, 272.f},
              vec3_t{82.f, 165.f, 225.f},
              vec3_t{82.f, 0, 225.f},
              white_index);
        
    //tall block
    push_face(scene,
              vec3_t{423.f, 330.f, 247.f},
              vec3_t{265.f, 330.f, 296.f},
              vec3_t{314.f, 330.f, 456.f},
              vec3_t{472.f, 330.f, 406.f},
              white_index);
    
    push_face(scene,
              vec3_t{423.f, 0, 247.f},
              vec3_t{423.f, 330.f, 247.f},
              vec3_t{472.f, 330.f, 406.f},
              vec3_t{472.f, 0, 406.f},
              white_index);
    
    push_face(scene,
              vec3_t{472.f, 0, 406.f},
              vec3_t{472.f, 330.f, 406.f},
              vec3_t{314.f, 330.f, 456.f},
              vec3_t{314.f, 0, 456.f},
              white_index);
    
    push_face(scene,
              vec3_t{314.f, 0, 456.f},
              vec3_t{314.f, 330.f, 456.f},
              vec3_t{265.f, 330.f, 296.f},
              vec3_t{265.f, 0, 296.f},
              white_index);
    
    push_face(scene,
              vec3_t{265.f, 0, 296.f},
              vec3_t{265.f, 330.f, 296.f},
              vec3_t{423.f, 330.f, 247.f},
              vec3_t{423.f, 0, 247.f},
              white_index);
    
   
}

const int width = 1280;
const int height = 1280;

void write_pixel(float* image, size_t pixel_row_pitch, int x, int y, vec3_t color)
{
    image[y * pixel_row_pitch + x * 4 + 0] = color.x;
    image[y * pixel_row_pitch + x * 4 + 1] = color.y;
    image[y * pixel_row_pitch + x * 4 + 2] = color.z;
}

enum trace_method_t
{
    RT_SCALAR,
    RT_SIMD
};

void ray_trace_scene(const scene_t& scene, const camera_t& camera, float* output_image, int width, int height, int sample_count, int bounce_count, trace_method_t method)
{
    vec3_t right = tanf(camera.fov) * camera.right;
    vec3_t down = tanf(camera.fov) * camera.down;
    size_t row_pitch_in_pixel = width * 4;
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            vec3_t accumulated_color = vec3_t{0};
            for(int sample_index = 0; sample_index < sample_count; sample_index++) {
                uint32_t seed = (x * uint32_t(1973) + y * uint32_t(9277) + sample_index * uint32_t(26699)) | 1;
                
                ray_t ray;
                ray.origin = camera.position;
                
                ray.direction = camera.look_at +
                (2.f * x / width - 1.f) * right +
                (2.f * y / width - height/width) * down ;
                
                ray.direction = normalize(ray.direction);
                
                vec3_t result_color = vec3_t{0};
                vec3_t throughput = vec3_t{1.f, 1.f, 1.f};
                
                for(int bounce_index = 0; bounce_index < bounce_count; bounce_index++) {
                    
                    ray_scene_intersect_info_t intersect_info ;
                    
                    if(method == RT_SCALAR)
                        intersect_info = intersect(ray, scene);
                    else
                        intersect_info = intersect_simd(ray, scene);
                    
                    if(intersect_info.triangle_index < 0)
                        break;
         
                     
                    const triangle_t& hit_triangle = scene.triangle_list[intersect_info.triangle_index];
                    const material_t& material = scene.material_list[hit_triangle.material_id];
                    
                    
                    vec3_t triangle_normal = normal(hit_triangle);
                    
                    ray.origin = (ray.origin + ray.direction * intersect_info.triangle_intersect_info.t) + triangle_normal * 0.001f;
                    
                    vec3_t new_direction = {0};
                    
                    while(length(new_direction) == 0)
                        new_direction = triangle_normal + random_unit_vector(seed);
                    
                    ray.direction = normalize(new_direction);
                    
                    result_color += material.emissive * throughput;
                    throughput *= material.albedo;
                    
                }
                
                float alpha = 1.f / (sample_index + 1);
                
                accumulated_color = alpha * result_color + (1.f - alpha) * accumulated_color;
                    
            }
            write_pixel(output_image, row_pitch_in_pixel, x, y, accumulated_color);

        }
    }
}


#include <stdint.h>

int main(int argc, const char * argv[]) {
    
    int arg_index = 1;
    
    bool parse_arg_failed = false;
    
    int sample_count = 16;
    int bounce_count = 8;
    trace_method_t method = RT_SIMD;
    
    while(arg_index < argc) {
        if(strcmp(argv[arg_index], "--sample_count") == 0) {
            if( (arg_index + 1) == argc ) {
                parse_arg_failed = true;
                break;
            }
            arg_index++;
            sscanf(argv[arg_index], "%d", &sample_count);
            arg_index++;
            
            continue;
        } else if(strcmp(argv[arg_index], "--bounce_count") == 0) {
            if( (arg_index + 1) == argc ) {
                parse_arg_failed = true;
                break;
            }
            arg_index++;
            sscanf(argv[arg_index], "%d", &bounce_count);
            arg_index++;
            
            continue;
        } else if(strcmp(argv[arg_index], "--method") == 0) {
            if( (arg_index + 1) == argc ) {
                parse_arg_failed = true;
                break;
            }
            arg_index++;
            if(strcmp(argv[arg_index], "scalar") == 0 ) {
                method = RT_SCALAR;
            } else if(strcmp(argv[arg_index], "simd") == 0 ) {
                method = RT_SIMD;
            }
            arg_index++;
        }
        
        else {
            parse_arg_failed = true;
            break;
        }
        
    }

    
    if(parse_arg_failed) {
        printf("invalid argument\n");
        return -1;
    }
        
    
    float* output_image = (float*) malloc( size_t(width * height) * 4 * sizeof(float) );

    memset(output_image, 0, size_t(width * height) * 4 * sizeof(float));
        
    scene_t scene;
    construct_cornell_box_scene(&scene);
    
    transpose_scene_soa(&scene);
    
    camera_t camera;
    camera.look_at = vec3_t{0, 1.f, 0};
    camera.right = vec3_t{1.f, 0, 0};
    camera.down = cross(camera.look_at, camera.right);
    camera.position = vec3_t{278.f, -800.f, 273.f};
    camera.fov = PI / 8.f;
    

    
    ray_trace_scene(scene, camera, output_image, width, height, sample_count, bounce_count, method);
    
    stbi_write_hdr("out.hdr", width, height, 4, output_image);
    
    return 0;
}
