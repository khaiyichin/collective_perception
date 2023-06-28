/*
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007, 2010 Gerard Jungman, Brian Gough
 * Copyright (C) 2002, 2003 Jason H Stover.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <cmath>
#include <cstring>
#include "custom_beta_cdf_gsl.hpp"

static unsigned int tests = 0;
static unsigned int passed = 0;
static unsigned int failed = 0;

static unsigned int verbose = 1;

static void initialise(void)
{
    const char *p = getenv("GSL_TEST_VERBOSE");

    /* 0 = show failures only (we always want to see these) */
    /* 1 = show passes and failures */

    if (p == 0) /* environment variable is not set */
        return;

    if (*p == '\0') /* environment variable is empty */
        return;

    verbose = strtoul(p, 0, 0);

    return;
}

static void update(int s)
{
    tests++;

    if (s == 0)
    {
        passed++;
    }
    else
    {
        failed++;
    }
}

void gsl_test_rel(double result, double expected, double relative_error,
                  const char *test_description, ...)
{
    int status;

    if (!tests)
        initialise();

    /* Check for NaN vs inf vs number */

    if (std::isnan(result) || std::isnan(expected))
    {
        status = std::isnan(result) != std::isnan(expected);
    }
    else if (std::isinf(result) || std::isinf(expected))
    {
        status = std::isinf(result) != std::isinf(expected);
    }
    else if ((expected > 0 && expected < GSL_DBL_MIN) || (expected < 0 && expected > -(GSL_DBL_MIN)))
    {
        status = -1;
    }
    else if (expected != 0)
    {
        status = (fabs(result - expected) / fabs(expected) > relative_error);
    }
    else
    {
        status = (fabs(result) > relative_error);
    }

    update(status);

    if (status || verbose)
    {
        printf(status ? "FAIL: " : "PASS: ");

#if HAVE_VPRINTF
        {
            va_list ap;
#ifdef STDC_HEADERS
            va_start(ap, test_description);
#else
            va_start(ap);
#endif
            vprintf(test_description, ap);
            va_end(ap);
        }
#endif

        if (status == 0)
        {
            if (std::strlen(test_description) < 45)
            {
                printf(" (%g observed vs %g expected)", result, expected);
            }
            else
            {
                printf(" (%g obs vs %g exp)", result, expected);
            }
        }
        else
        {
            printf(" (%.18g observed vs %.18g expected)", result, expected);
        }

        if (status == -1)
        {
            printf(" [test uses subnormal value]");
        }

        if (status && !verbose)
            printf(" [%u]", tests);

        printf("\n");
        fflush(stdout);
    }
}

void gsl_test_rel(double result, double expected, double relative_error,
                  const char *test_description, ...);

#define TEST(func, args, value, tol)                \
    {                                               \
        double res = func args;                     \
        gsl_test_rel(res, value, tol, #func #args); \
    };
#define TEST_TOL0 (2.0 * GSL_DBL_EPSILON)
#define TEST_TOL1 (16.0 * GSL_DBL_EPSILON)
#define TEST_TOL2 (256.0 * GSL_DBL_EPSILON)
#define TEST_TOL3 (2048.0 * GSL_DBL_EPSILON)
#define TEST_TOL4 (16384.0 * GSL_DBL_EPSILON)
#define TEST_TOL5 (131072.0 * GSL_DBL_EPSILON)
#define TEST_TOL6 (1048576.0 * GSL_DBL_EPSILON)

void test_beta()
{
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.0, 1.2, 1.3), 0.0, 0.0);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (1e-100, 1.2, 1.3), 1.34434944656489596e-120, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.001, 1.2, 1.3), 3.37630042504535813e-4, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.01, 1.2, 1.3), 5.34317264038929473e-3, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.1, 1.2, 1.3), 8.33997828306748346e-2, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.325, 1.2, 1.3), 3.28698654180583916e-1, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.5, 1.2, 1.3), 5.29781429451299081e-1, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.9, 1.2, 1.3), 9.38529397224430659e-1, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.99, 1.2, 1.3), 9.96886438341254380e-1, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.999, 1.2, 1.3), 9.99843792833067634e-1, TEST_TOL6);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (1.0, 1.2, 1.3), 1.0, TEST_TOL6);

    // alpha = 23, beta = 19
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.3, 23, 19), 0.00044842294043878698192, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.35, 23, 19), 0.0045840996678660188984, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.356, 23, 19), 0.0058241095815739895442, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.3561, 23, 19), 0.0058470130849443387525, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.35614, 23, 19), 0.0058561962034902881324, TEST_TOL4);

    // alpha = 31, beta = 79
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.3, 31, 79), 0.67308297856070575804, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.37, 31, 79), 0.97616512540601341197, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.370, 31, 79), 0.97616512540601341197, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.3705, 31, 79), 0.97675505532397843833, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.37058, 31, 79), 0.97684827595247680776, TEST_TOL4);

    // alpha = 601, beta = 594
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.5, 601, 594), 0.41973709344382181818, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.52, 601, 594), 0.88108300669454875376, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.529, 601, 594), 0.96435531068564928070, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.5297, 601, 594), 0.96799702212233684762, TEST_TOL4);
    TEST(CustomBetaCdfGSL::gsl_cdf_beta_P, (0.52978, 601, 594), 0.96839294958567923022, TEST_TOL4);

    std::cout << std::endl
              << "########## TESTING COMPLETE ##########" << std::endl;
}

int main()
{
    // Run tests
    test_beta();

    return 0;
}