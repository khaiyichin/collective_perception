#include "custom_beta_cdf_gsl.hpp"

namespace CustomBetaCdfGSL
{
    /****************************************/
    /* error.c */
    /****************************************/

    void gsl_error(const char *reason, const char *file, int line, int gsl_errno)
    {
        /* Alternative error output instead of the GSL vanilla output.
         * This is so that we can avoid using global variables since they're all
         * defined in the same file.
         * Fixed: Khai Yi Chin khaiyichin@gmail.com 06/27/2023 10:56 EST
         */
        // if (gsl_error_handler)
        // {
        //     (*gsl_error_handler)(reason, file, line, gsl_errno);
        //     return;
        // }

        // gsl_stream_printf("ERROR", file, line, reason);
        std::cout << "ERROR" << file << " " << line << " " << reason << std::endl;

        fflush(stdout);
        fprintf(stderr, "Default GSL error handler invoked.\n");
        fflush(stderr);

        abort();
    }

    /****************************************/
    /* erfc.c */
    /****************************************/

    double erfc8_sum(const double &x)
    {
        /* estimates erfc(x) valid for 8 < x < 100 */
        /* This is based on index 5725 in Hart et al */

        static double P[] = {
            2.97886562639399288862,
            7.409740605964741794425,
            6.1602098531096305440906,
            5.019049726784267463450058,
            1.275366644729965952479585264,
            0.5641895835477550741253201704};
        static double Q[] = {
            3.3690752069827527677,
            9.608965327192787870698,
            17.08144074746600431571095,
            12.0489519278551290360340491,
            9.396034016235054150430579648,
            2.260528520767326969591866945,
            1.0};
        double num = 0.0, den = 0.0;
        int i;

        num = P[5];
        for (i = 4; i >= 0; --i)
        {
            num = x * num + P[i];
        }
        den = Q[6];
        for (i = 5; i >= 0; --i)
        {
            den = x * den + Q[i];
        }

        return num / den;
    }

    double erfc8(const double &x)
    {
        double e;
        e = erfc8_sum(x);
        e *= exp(-x * x);
        return e;
    }

    /*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_erfc_e(const double &x, gsl_sf_result *result)
    {
        const double ax = fabs(x);
        double e_val, e_err;

        /* CHECK_POINTER(result) */

        if (ax <= 1.0)
        {
            double t = 2.0 * ax - 1.0;
            gsl_sf_result c;
            cheb_eval_e(&erfc_xlt1_cs, t, &c);
            e_val = c.val;
            e_err = c.err;
        }
        else if (ax <= 5.0)
        {
            double ex2 = exp(-x * x);
            double t = 0.5 * (ax - 3.0);
            gsl_sf_result c;
            cheb_eval_e(&erfc_x15_cs, t, &c);
            e_val = ex2 * c.val;
            e_err = ex2 * (c.err + 2.0 * fabs(x) * GSL_DBL_EPSILON);
        }
        else if (ax < 10.0)
        {
            double exterm = exp(-x * x) / ax;
            double t = (2.0 * ax - 15.0) / 5.0;
            gsl_sf_result c;
            cheb_eval_e(&erfc_x510_cs, t, &c);
            e_val = exterm * c.val;
            e_err = exterm * (c.err + 2.0 * fabs(x) * GSL_DBL_EPSILON + GSL_DBL_EPSILON);
        }
        else
        {
            e_val = erfc8(ax);
            e_err = (x * x + 1.0) * GSL_DBL_EPSILON * fabs(e_val);
        }

        if (x < 0.0)
        {
            result->val = 2.0 - e_val;
            result->err = e_err;
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
        }
        else
        {
            result->val = e_val;
            result->err = e_err;
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
        }

        return GSL_SUCCESS;
    }

    /****************************************/
    /* zeta.c */
    /****************************************/

    /*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_hzeta_e(const double &s, const double &q, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (s <= 1.0 || q <= 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else
        {
            const double max_bits = 54.0;
            const double ln_term0 = -s * log(q);

            if (ln_term0 < GSL_LOG_DBL_MIN + 1.0)
            {
                UNDERFLOW_ERROR(result);
            }
            else if (ln_term0 > GSL_LOG_DBL_MAX - 1.0)
            {
                OVERFLOW_ERROR(result);
            }
            else if ((s > max_bits && q < 1.0) || (s > 0.5 * max_bits && q < 0.25))
            {
                result->val = pow(q, -s);
                result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
                return GSL_SUCCESS;
            }
            else if (s > 0.5 * max_bits && q < 1.0)
            {
                const double p1 = pow(q, -s);
                const double p2 = pow(q / (1.0 + q), s);
                const double p3 = pow(q / (2.0 + q), s);
                result->val = p1 * (1.0 + p2 + p3);
                result->err = GSL_DBL_EPSILON * (0.5 * s + 2.0) * fabs(result->val);
                return GSL_SUCCESS;
            }
            else
            {
                /* Euler-Maclaurin summation formula
                 * [Moshier, p. 400, with several typo corrections]
                 */
                const int jmax = 12;
                const int kmax = 10;
                int j, k;
                const double pmax = pow(kmax + q, -s);
                double scp = s;
                double pcp = pmax / (kmax + q);
                double ans = pmax * ((kmax + q) / (s - 1.0) + 0.5);

                for (k = 0; k < kmax; k++)
                {
                    ans += pow(k + q, -s);
                }

                for (j = 0; j <= jmax; j++)
                {
                    double delta = hzeta_c[j + 1] * scp * pcp;
                    ans += delta;
                    if (fabs(delta / ans) < 0.5 * GSL_DBL_EPSILON)
                        break;
                    scp *= (s + 2 * j + 1) * (s + 2 * j + 2);
                    pcp /= (kmax + q) * (kmax + q);
                }

                result->val = ans;
                result->err = 2.0 * (jmax + 1.0) * GSL_DBL_EPSILON * fabs(ans);
                return GSL_SUCCESS;
            }
        }
    }

    /****************************************/
    /* psi.c */
    /****************************************/

    /* digamma for x both positive and negative; we do both
     * cases here because of the way we use even/odd parts
     * of the function
     */
    int psi_x(const double x, gsl_sf_result *result)
    {
        const double y = fabs(x);

        if (x == 0.0 || x == -1.0 || x == -2.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (y >= 2.0)
        {
            const double t = 8.0 / (y * y) - 1.0;
            gsl_sf_result result_c;
            cheb_eval_e(&apsi_cs, t, &result_c);
            if (x < 0.0)
            {
                const double s = sin(M_PI * x);
                const double c = cos(M_PI * x);
                if (fabs(s) < 2.0 * GSL_SQRT_DBL_MIN)
                {
                    DOMAIN_ERROR(result);
                }
                else
                {
                    result->val = log(y) - 0.5 / x + result_c.val - M_PI * c / s;
                    result->err = M_PI * fabs(x) * GSL_DBL_EPSILON / (s * s);
                    result->err += result_c.err;
                    result->err += GSL_DBL_EPSILON * fabs(result->val);
                    return GSL_SUCCESS;
                }
            }
            else
            {
                result->val = log(y) - 0.5 / x + result_c.val;
                result->err = result_c.err;
                result->err += GSL_DBL_EPSILON * fabs(result->val);
                return GSL_SUCCESS;
            }
        }
        else
        { /* -2 < x < 2 */
            gsl_sf_result result_c;

            if (x < -1.0)
            { /* x = -2 + v */
                const double v = x + 2.0;
                const double t1 = 1.0 / x;
                const double t2 = 1.0 / (x + 1.0);
                const double t3 = 1.0 / v;
                cheb_eval_e(&psi_cs, 2.0 * v - 1.0, &result_c);

                result->val = -(t1 + t2 + t3) + result_c.val;
                result->err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2 * t2)) + fabs(x / (t3 * t3)));
                result->err += result_c.err;
                result->err += GSL_DBL_EPSILON * fabs(result->val);
                return GSL_SUCCESS;
            }
            else if (x < 0.0)
            { /* x = -1 + v */
                const double v = x + 1.0;
                const double t1 = 1.0 / x;
                const double t2 = 1.0 / v;
                cheb_eval_e(&psi_cs, 2.0 * v - 1.0, &result_c);

                result->val = -(t1 + t2) + result_c.val;
                result->err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2 * t2)));
                result->err += result_c.err;
                result->err += GSL_DBL_EPSILON * fabs(result->val);
                return GSL_SUCCESS;
            }
            else if (x < 1.0)
            { /* x = v */
                const double t1 = 1.0 / x;
                cheb_eval_e(&psi_cs, 2.0 * x - 1.0, &result_c);

                result->val = -t1 + result_c.val;
                result->err = GSL_DBL_EPSILON * t1;
                result->err += result_c.err;
                result->err += GSL_DBL_EPSILON * fabs(result->val);
                return GSL_SUCCESS;
            }
            else
            { /* x = 1 + v */
                const double v = x - 1.0;
                return cheb_eval_e(&psi_cs, 2.0 * v - 1.0, result);
            }
        }
    }

    /* generic polygamma; assumes n >= 0 and x > 0
     */
    int psi_n_xg0(const int n, const double x, gsl_sf_result *result)
    {
        if (n == 0)
        {
            return gsl_sf_psi_e(x, result);
        }
        else
        {
            /* Abramowitz + Stegun 6.4.10 */
            gsl_sf_result ln_nf;
            gsl_sf_result hzeta;
            int stat_hz = gsl_sf_hzeta_e(n + 1.0, x, &hzeta);
            int stat_nf = gsl_sf_lnfact_e((unsigned int)n, &ln_nf);
            int stat_e = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
                                               hzeta.val, hzeta.err,
                                               result);
            if (GSL_IS_EVEN(n))
                result->val = -result->val;
            return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
        }
    }

    /*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_psi_e(const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */
        return psi_x(x, result);
    }

    int gsl_sf_psi_int_e(const int n, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (n <= 0)
        {
            DOMAIN_ERROR(result);
        }
        else if (n <= PSI_TABLE_NMAX)
        {
            result->val = psi_table[n];
            result->err = GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else
        {
            /* Abramowitz+Stegun 6.3.18 */
            const double c2 = -1.0 / 12.0;
            const double c3 = 1.0 / 120.0;
            const double c4 = -1.0 / 252.0;
            const double c5 = 1.0 / 240.0;
            const double ni2 = (1.0 / n) * (1.0 / n);
            const double ser = ni2 * (c2 + ni2 * (c3 + ni2 * (c4 + ni2 * c5)));
            result->val = log(n) - 0.5 / n + ser;
            result->err = GSL_DBL_EPSILON * (fabs(log(n)) + fabs(0.5 / n) + fabs(ser));
            result->err += GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
    }

    int gsl_sf_psi_1_int_e(const int n, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */
        if (n <= 0)
        {
            DOMAIN_ERROR(result);
        }
        else if (n <= PSI_1_TABLE_NMAX)
        {
            result->val = psi_1_table[n];
            result->err = GSL_DBL_EPSILON * result->val;
            return GSL_SUCCESS;
        }
        else
        {
            /* Abramowitz+Stegun 6.4.12
             * double-precision for n > 100
             */
            const double c0 = -1.0 / 30.0;
            const double c1 = 1.0 / 42.0;
            const double c2 = -1.0 / 30.0;
            const double ni2 = (1.0 / n) * (1.0 / n);
            const double ser = ni2 * ni2 * (c0 + ni2 * (c1 + c2 * ni2));
            result->val = (1.0 + 0.5 / n + 1.0 / (6.0 * n * n) + ser) / n;
            result->err = GSL_DBL_EPSILON * result->val;
            return GSL_SUCCESS;
        }
    }

    int gsl_sf_psi_1_e(const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (x == 0.0 || x == -1.0 || x == -2.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (x > 0.0)
        {
            return psi_n_xg0(1, x, result);
        }
        else if (x > -5.0)
        {
            /* Abramowitz + Stegun 6.4.6 */
            int M = -floor(x);
            double fx = x + M;
            double sum = 0.0;
            int m;

            if (fx == 0.0)
                DOMAIN_ERROR(result);

            for (m = 0; m < M; ++m)
                sum += 1.0 / ((x + m) * (x + m));

            {
                int stat_psi = psi_n_xg0(1, fx, result);
                result->val += sum;
                result->err += M * GSL_DBL_EPSILON * sum;
                return stat_psi;
            }
        }
        else
        {
            /* Abramowitz + Stegun 6.4.7 */
            const double sin_px = sin(M_PI * x);
            const double d = M_PI * M_PI / (sin_px * sin_px);
            gsl_sf_result r;
            int stat_psi = psi_n_xg0(1, 1.0 - x, &r);
            result->val = d - r.val;
            result->err = r.err + 2.0 * GSL_DBL_EPSILON * d;
            return stat_psi;
        }
    }

    int gsl_sf_psi_n_e(const int n, const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (n == 0)
        {
            return gsl_sf_psi_e(x, result);
        }
        else if (n == 1)
        {
            return gsl_sf_psi_1_e(x, result);
        }
        else if (n < 0 || x <= 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else
        {
            gsl_sf_result ln_nf;
            gsl_sf_result hzeta;
            int stat_hz = gsl_sf_hzeta_e(n + 1.0, x, &hzeta);
            int stat_nf = gsl_sf_lnfact_e((unsigned int)n, &ln_nf);
            int stat_e = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
                                               hzeta.val, hzeta.err,
                                               result);
            if (GSL_IS_EVEN(n))
                result->val = -result->val;
            return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
        }
    }

    /****************************************/
    /* cheb.c */
    /****************************************/

    int cheb_eval_e(const cheb_series *cs,
                    const double x,
                    gsl_sf_result *result)
    {
        int j;
        double d = 0.0;
        double dd = 0.0;

        double y = (2.0 * x - cs->a - cs->b) / (cs->b - cs->a);
        double y2 = 2.0 * y;

        double e = 0.0;

        for (j = cs->order; j >= 1; j--)
        {
            double temp = d;
            d = y2 * d - dd + cs->c[j];
            e += fabs(y2 * temp) + fabs(dd) + fabs(cs->c[j]);
            dd = temp;
        }

        {
            double temp = d;
            d = y * d - dd + 0.5 * cs->c[0];
            e += fabs(y * temp) + fabs(dd) + 0.5 * fabs(cs->c[0]);
        }

        result->val = d;
        result->err = GSL_DBL_EPSILON * e + fabs(cs->c[cs->order]);

        return GSL_SUCCESS;
    }

    /****************************************/
    /* log.c */
    /****************************************/

    /*-*-*-*-*-*-*-*-*-*-*-* Private Section *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_log_1plusx_e(const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (x <= -1.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (fabs(x) < GSL_ROOT6_DBL_EPSILON)
        {
            const double c1 = -0.5;
            const double c2 = 1.0 / 3.0;
            const double c3 = -1.0 / 4.0;
            const double c4 = 1.0 / 5.0;
            const double c5 = -1.0 / 6.0;
            const double c6 = 1.0 / 7.0;
            const double c7 = -1.0 / 8.0;
            const double c8 = 1.0 / 9.0;
            const double c9 = -1.0 / 10.0;
            const double t = c5 + x * (c6 + x * (c7 + x * (c8 + x * c9)));
            result->val = x * (1.0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * t)))));
            result->err = GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else if (fabs(x) < 0.5)
        {
            double t = 0.5 * (8.0 * x + 1.0) / (x + 2.0);
            gsl_sf_result c;
            cheb_eval_e(&lopx_cs, t, &c);
            result->val = x * c.val;
            result->err = fabs(x * c.err);
            return GSL_SUCCESS;
        }
        else
        {
            result->val = log(1.0 + x);
            result->err = GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
    }

    int gsl_sf_log_1plusx_mx_e(const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (x <= -1.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (fabs(x) < GSL_ROOT5_DBL_EPSILON)
        {
            const double c1 = -0.5;
            const double c2 = 1.0 / 3.0;
            const double c3 = -1.0 / 4.0;
            const double c4 = 1.0 / 5.0;
            const double c5 = -1.0 / 6.0;
            const double c6 = 1.0 / 7.0;
            const double c7 = -1.0 / 8.0;
            const double c8 = 1.0 / 9.0;
            const double c9 = -1.0 / 10.0;
            const double t = c5 + x * (c6 + x * (c7 + x * (c8 + x * c9)));
            result->val = x * x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * t))));
            result->err = GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else if (fabs(x) < 0.5)
        {
            double t = 0.5 * (8.0 * x + 1.0) / (x + 2.0);
            gsl_sf_result c;
            cheb_eval_e(&lopxmx_cs, t, &c);
            result->val = x * x * c.val;
            result->err = x * x * c.err;
            return GSL_SUCCESS;
        }
        else
        {
            const double lterm = log(1.0 + x);
            result->val = lterm - x;
            result->err = GSL_DBL_EPSILON * (fabs(lterm) + fabs(x));
            return GSL_SUCCESS;
        }
    }

    /****************************************/
    /* exp.c */
    /****************************************/

    /* Evaluate the continued fraction for exprel.
     * [Abramowitz+Stegun, 4.2.41]
     */
    int exprel_n_CF(const double N, const double x, gsl_sf_result *result)
    {
        const double RECUR_BIG = GSL_SQRT_DBL_MAX;
        const int maxiter = 5000;
        int n = 1;
        double Anm2 = 1.0;
        double Bnm2 = 0.0;
        double Anm1 = 0.0;
        double Bnm1 = 1.0;
        double a1 = 1.0;
        double b1 = 1.0;
        double a2 = -x;
        double b2 = N + 1;
        double an, bn;

        double fn;

        double An = b1 * Anm1 + a1 * Anm2; /* A1 */
        double Bn = b1 * Bnm1 + a1 * Bnm2; /* B1 */

        /* One explicit step, before we get to the main pattern. */
        n++;
        Anm2 = Anm1;
        Bnm2 = Bnm1;
        Anm1 = An;
        Bnm1 = Bn;
        An = b2 * Anm1 + a2 * Anm2; /* A2 */
        Bn = b2 * Bnm1 + a2 * Bnm2; /* B2 */

        fn = An / Bn;

        while (n < maxiter)
        {
            double old_fn;
            double del;
            n++;
            Anm2 = Anm1;
            Bnm2 = Bnm1;
            Anm1 = An;
            Bnm1 = Bn;
            an = (GSL_IS_ODD(n) ? ((n - 1) / 2) * x : -(N + (n / 2) - 1) * x);
            bn = N + n - 1;
            An = bn * Anm1 + an * Anm2;
            Bn = bn * Bnm1 + an * Bnm2;

            if (fabs(An) > RECUR_BIG || fabs(Bn) > RECUR_BIG)
            {
                An /= RECUR_BIG;
                Bn /= RECUR_BIG;
                Anm1 /= RECUR_BIG;
                Bnm1 /= RECUR_BIG;
                Anm2 /= RECUR_BIG;
                Bnm2 /= RECUR_BIG;
            }

            old_fn = fn;
            fn = An / Bn;
            del = old_fn / fn;

            if (fabs(del - 1.0) < 2.0 * GSL_DBL_EPSILON)
                break;
        }

        result->val = fn;
        result->err = 4.0 * (n + 1.0) * GSL_DBL_EPSILON * fabs(fn);

        if (n == maxiter)
            GSL_ERROR("error", GSL_EMAXITER);
        else
            return GSL_SUCCESS;
    }

    int gsl_sf_exp_mult_err_e(const double x, const double dx,
                              const double y, const double dy,
                              gsl_sf_result *result)
    {
        const double ay = fabs(y);

        if (y == 0.0)
        {
            result->val = 0.0;
            result->err = fabs(dy * exp(x));
            return GSL_SUCCESS;
        }
        else if ((x < 0.5 * GSL_LOG_DBL_MAX && x > 0.5 * GSL_LOG_DBL_MIN) && (ay < 0.8 * GSL_SQRT_DBL_MAX && ay > 1.2 * GSL_SQRT_DBL_MIN))
        {
            double ex = exp(x);
            result->val = y * ex;
            result->err = ex * (fabs(dy) + fabs(y * dx));
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else
        {
            const double ly = log(ay);
            const double lnr = x + ly;

            if (lnr > GSL_LOG_DBL_MAX - 0.01)
            {
                OVERFLOW_ERROR(result);
            }
            else if (lnr < GSL_LOG_DBL_MIN + 0.01)
            {
                UNDERFLOW_ERROR(result);
            }
            else
            {
                const double sy = GSL_SIGN(y);
                const double M = floor(x);
                const double N = floor(ly);
                const double a = x - M;
                const double b = ly - N;
                const double eMN = exp(M + N);
                const double eab = exp(a + b);
                result->val = sy * eMN * eab;
                result->err = eMN * eab * 2.0 * GSL_DBL_EPSILON;
                result->err += eMN * eab * fabs(dy / y);
                result->err += eMN * eab * fabs(dx);
                return GSL_SUCCESS;
            }
        }
    }

    int gsl_sf_exprel_n_CF_e(const double N, const double x, gsl_sf_result *result)
    {
        return exprel_n_CF(N, x, result);
    }

    int gsl_sf_exp_err_e(const double x, const double dx, gsl_sf_result *result)
    {
        const double adx = fabs(dx);

        /* CHECK_POINTER(result) */

        if (x + adx > GSL_LOG_DBL_MAX)
        {
            OVERFLOW_ERROR(result);
        }
        else if (x - adx < GSL_LOG_DBL_MIN)
        {
            UNDERFLOW_ERROR(result);
        }
        else
        {
            const double ex = exp(x);
            const double edx = exp(adx);
            result->val = ex;
            result->err = ex * GSL_MAX_DBL(GSL_DBL_EPSILON, edx - 1.0 / edx);
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
    }

    /****************************************/
    /* gamma.c */
    /****************************************/

    /*-*-*-*-*-*-*-*-*-*-*-* Private Section *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_lnfact_e(const unsigned int n, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (n <= GSL_SF_FACT_NMAX)
        {
            result->val = log(fact_table[n].f);
            result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else
        {
            gsl_sf_lngamma_e(n + 1.0, result);
            return GSL_SUCCESS;
        }
    }

    /* x near a negative integer
     * Calculates sign as well as log(|gamma(x)|).
     * x = -N + eps
     * assumes N >= 1
     */
    int lngamma_sgn_sing(int N, double eps, gsl_sf_result *lng, double *sgn)
    {
        if (eps == 0.0)
        {
            lng->val = 0.0;
            lng->err = 0.0;
            *sgn = 0.0;
            GSL_ERROR("error", GSL_EDOM);
        }
        else if (N == 1)
        {
            /* calculate series for
             * g = eps gamma(-1+eps) + 1 + eps/2 (1+3eps)/(1-eps^2)
             * double-precision for |eps| < 0.02
             */
            const double c0 = 0.07721566490153286061;
            const double c1 = 0.08815966957356030521;
            const double c2 = -0.00436125434555340577;
            const double c3 = 0.01391065882004640689;
            const double c4 = -0.00409427227680839100;
            const double c5 = 0.00275661310191541584;
            const double c6 = -0.00124162645565305019;
            const double c7 = 0.00065267976121802783;
            const double c8 = -0.00032205261682710437;
            const double c9 = 0.00016229131039545456;
            const double g5 = c5 + eps * (c6 + eps * (c7 + eps * (c8 + eps * c9)));
            const double g = eps * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + eps * (c4 + eps * g5)))));

            /* calculate eps gamma(-1+eps), a negative quantity */
            const double gam_e = g - 1.0 - 0.5 * eps * (1.0 + 3.0 * eps) / (1.0 - eps * eps);

            lng->val = log(fabs(gam_e) / fabs(eps));
            lng->err = 2.0 * GSL_DBL_EPSILON * fabs(lng->val);
            *sgn = (eps > 0.0 ? -1.0 : 1.0);
            return GSL_SUCCESS;
        }
        else
        {
            double g;

            /* series for sin(Pi(N+1-eps))/(Pi eps) modulo the sign
             * double-precision for |eps| < 0.02
             */
            const double cs1 = -1.6449340668482264365;
            const double cs2 = 0.8117424252833536436;
            const double cs3 = -0.1907518241220842137;
            const double cs4 = 0.0261478478176548005;
            const double cs5 = -0.0023460810354558236;
            const double e2 = eps * eps;
            const double sin_ser = 1.0 + e2 * (cs1 + e2 * (cs2 + e2 * (cs3 + e2 * (cs4 + e2 * cs5))));

            /* calculate series for ln(gamma(1+N-eps))
             * double-precision for |eps| < 0.02
             */
            double aeps = fabs(eps);
            double c1, c2, c3, c4, c5, c6, c7;
            double lng_ser;
            gsl_sf_result c0;
            gsl_sf_result psi_0;
            gsl_sf_result psi_1;
            gsl_sf_result psi_2;
            gsl_sf_result psi_3;
            gsl_sf_result psi_4;
            gsl_sf_result psi_5;
            gsl_sf_result psi_6;
            psi_2.val = 0.0;
            psi_3.val = 0.0;
            psi_4.val = 0.0;
            psi_5.val = 0.0;
            psi_6.val = 0.0;
            gsl_sf_lnfact_e(N, &c0);
            gsl_sf_psi_int_e(N + 1, &psi_0);
            gsl_sf_psi_1_int_e(N + 1, &psi_1);
            if (aeps > 0.00001)
                gsl_sf_psi_n_e(2, N + 1.0, &psi_2);
            if (aeps > 0.0002)
                gsl_sf_psi_n_e(3, N + 1.0, &psi_3);
            if (aeps > 0.001)
                gsl_sf_psi_n_e(4, N + 1.0, &psi_4);
            if (aeps > 0.005)
                gsl_sf_psi_n_e(5, N + 1.0, &psi_5);
            if (aeps > 0.01)
                gsl_sf_psi_n_e(6, N + 1.0, &psi_6);
            c1 = psi_0.val;
            c2 = psi_1.val / 2.0;
            c3 = psi_2.val / 6.0;
            c4 = psi_3.val / 24.0;
            c5 = psi_4.val / 120.0;
            c6 = psi_5.val / 720.0;
            c7 = psi_6.val / 5040.0;
            lng_ser = c0.val - eps * (c1 - eps * (c2 - eps * (c3 - eps * (c4 - eps * (c5 - eps * (c6 - eps * c7))))));

            /* calculate
             * g = ln(|eps gamma(-N+eps)|)
             *   = -ln(gamma(1+N-eps)) + ln(|eps Pi/sin(Pi(N+1+eps))|)
             */
            g = -lng_ser - log(sin_ser);

            lng->val = g - log(fabs(eps));
            lng->err = c0.err + 2.0 * GSL_DBL_EPSILON * (fabs(g) + fabs(lng->val));

            *sgn = (GSL_IS_ODD(N) ? -1.0 : 1.0) * (eps > 0.0 ? 1.0 : -1.0);

            return GSL_SUCCESS;
        }
    }

    /* Lanczos method for real x > 0;
     * gamma=7, truncated at 1/(z+8)
     * [J. SIAM Numer. Anal, Ser. B, 1 (1964) 86]
     */
    int lngamma_lanczos(double x, gsl_sf_result *result)
    {
        int k;
        double Ag;
        double term1, term2;

        x -= 1.0; /* Lanczos writes z! instead of Gamma(z) */

        Ag = lanczos_7_c[0];
        for (k = 1; k <= 8; k++)
        {
            Ag += lanczos_7_c[k] / (x + k);
        }

        /* (x+0.5)*log(x+7.5) - (x+7.5) + LogRootTwoPi_ + log(Ag(x)) */
        term1 = (x + 0.5) * log((x + 7.5) / M_E);
        term2 = LogRootTwoPi_ + log(Ag);
        result->val = term1 + (term2 - 7.0);
        result->err = 2.0 * GSL_DBL_EPSILON * (fabs(term1) + fabs(term2) + 7.0);
        result->err += GSL_DBL_EPSILON * fabs(result->val);

        return GSL_SUCCESS;
    }

    int lngamma_1_pade(const double eps, gsl_sf_result *result)
    {
        /* Use (2,2) Pade for Log[Gamma[1+eps]]/eps
         * plus a correction series.
         */
        const double n1 = -1.0017419282349508699871138440;
        const double n2 = 1.7364839209922879823280541733;
        const double d1 = 1.2433006018858751556055436011;
        const double d2 = 5.0456274100274010152489597514;
        const double num = (eps + n1) * (eps + n2);
        const double den = (eps + d1) * (eps + d2);
        const double pade = 2.0816265188662692474880210318 * num / den;
        const double c0 = 0.004785324257581753;
        const double c1 = -0.01192457083645441;
        const double c2 = 0.01931961413960498;
        const double c3 = -0.02594027398725020;
        const double c4 = 0.03141928755021455;
        const double eps5 = eps * eps * eps * eps * eps;
        const double corr = eps5 * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + c4 * eps))));
        result->val = eps * (pade + corr);
        result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
        return GSL_SUCCESS;
    }

    int lngamma_2_pade(const double eps, gsl_sf_result *result)
    {
        /* Use (2,2) Pade for Log[Gamma[2+eps]]/eps
         * plus a correction series.
         */
        const double n1 = 1.000895834786669227164446568;
        const double n2 = 4.209376735287755081642901277;
        const double d1 = 2.618851904903217274682578255;
        const double d2 = 10.85766559900983515322922936;
        const double num = (eps + n1) * (eps + n2);
        const double den = (eps + d1) * (eps + d2);
        const double pade = 2.85337998765781918463568869 * num / den;
        const double c0 = 0.0001139406357036744;
        const double c1 = -0.0001365435269792533;
        const double c2 = 0.0001067287169183665;
        const double c3 = -0.0000693271800931282;
        const double c4 = 0.0000407220927867950;
        const double eps5 = eps * eps * eps * eps * eps;
        const double corr = eps5 * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + c4 * eps))));
        result->val = eps * (pade + corr);
        result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
        return GSL_SUCCESS;
    }

    /* x = eps near zero
     * gives double-precision for |eps| < 0.02
     */
    int lngamma_sgn_0(double eps, gsl_sf_result *lng, double *sgn)
    {
        /* calculate series for g(eps) = Gamma(eps) eps - 1/(1+eps) - eps/2 */
        const double c1 = -0.07721566490153286061;
        const double c2 = -0.01094400467202744461;
        const double c3 = 0.09252092391911371098;
        const double c4 = -0.01827191316559981266;
        const double c5 = 0.01800493109685479790;
        const double c6 = -0.00685088537872380685;
        const double c7 = 0.00399823955756846603;
        const double c8 = -0.00189430621687107802;
        const double c9 = 0.00097473237804513221;
        const double c10 = -0.00048434392722255893;
        const double g6 = c6 + eps * (c7 + eps * (c8 + eps * (c9 + eps * c10)));
        const double g = eps * (c1 + eps * (c2 + eps * (c3 + eps * (c4 + eps * (c5 + eps * g6)))));

        /* calculate Gamma(eps) eps, a positive quantity */
        const double gee = g + 1.0 / (1.0 + eps) + 0.5 * eps;

        lng->val = log(gee / fabs(eps));
        lng->err = 4.0 * GSL_DBL_EPSILON * fabs(lng->val);
        *sgn = GSL_SIGN(eps);

        return GSL_SUCCESS;
    }

    int gsl_sf_gammastar_e(const double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (x <= 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (x < 0.5)
        {
            gsl_sf_result lg;
            const int stat_lg = gsl_sf_lngamma_e(x, &lg);
            const double lx = log(x);
            const double c = 0.5 * (M_LN2 + M_LNPI);
            const double lnr_val = lg.val - (x - 0.5) * lx + x - c;
            const double lnr_err = lg.err + 2.0 * GSL_DBL_EPSILON * ((x + 0.5) * fabs(lx) + c);
            const int stat_e = gsl_sf_exp_err_e(lnr_val, lnr_err, result);
            return GSL_ERROR_SELECT_2(stat_lg, stat_e);
        }
        else if (x < 2.0)
        {
            const double t = 4.0 / 3.0 * (x - 0.5) - 1.0;
            return cheb_eval_e(&gstar_a_cs, t, result);
        }
        else if (x < 10.0)
        {
            const double t = 0.25 * (x - 2.0) - 1.0;
            gsl_sf_result c;
            cheb_eval_e(&gstar_b_cs, t, &c);
            result->val = c.val / (x * x) + 1.0 + 1.0 / (12.0 * x);
            result->err = c.err / (x * x);
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else if (x < 1.0 / GSL_ROOT4_DBL_EPSILON)
        {
            return gammastar_ser(x, result);
        }
        else if (x < 1.0 / GSL_DBL_EPSILON)
        {
            /* Use Stirling formula for Gamma(x).
             */
            const double xi = 1.0 / x;
            result->val = 1.0 + xi / 12.0 * (1.0 + xi / 24.0 * (1.0 - xi * (139.0 / 180.0 + 571.0 / 8640.0 * xi)));
            result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_SUCCESS;
        }
        else
        {
            result->val = 1.0;
            result->err = 1.0 / x;
            return GSL_SUCCESS;
        }
    }

    /* series for gammastar(x)
     * double-precision for x > 10.0
     */
    static int
    gammastar_ser(const double x, gsl_sf_result *result)
    {
        /* Use the Stirling series for the correction to Log(Gamma(x)),
         * which is better behaved and easier to compute than the
         * regular Stirling series for Gamma(x).
         */
        const double y = 1.0 / (x * x);
        const double c0 = 1.0 / 12.0;
        const double c1 = -1.0 / 360.0;
        const double c2 = 1.0 / 1260.0;
        const double c3 = -1.0 / 1680.0;
        const double c4 = 1.0 / 1188.0;
        const double c5 = -691.0 / 360360.0;
        const double c6 = 1.0 / 156.0;
        const double c7 = -3617.0 / 122400.0;
        const double ser = c0 + y * (c1 + y * (c2 + y * (c3 + y * (c4 + y * (c5 + y * (c6 + y * c7))))));
        result->val = exp(ser / x);
        result->err = 2.0 * GSL_DBL_EPSILON * result->val * GSL_MAX_DBL(1.0, ser / x);
        return GSL_SUCCESS;
    }

    /*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_lngamma_e(double x, gsl_sf_result *result)
    {
        /* CHECK_POINTER(result) */

        if (fabs(x - 1.0) < 0.01)
        {
            /* Note that we must amplify the errors
             * from the Pade evaluations because of
             * the way we must pass the argument, i.e.
             * writing (1-x) is a loss of precision
             * when x is near 1.
             */
            int stat = lngamma_1_pade(x - 1.0, result);
            result->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 1.0));
            return stat;
        }
        else if (fabs(x - 2.0) < 0.01)
        {
            int stat = lngamma_2_pade(x - 2.0, result);
            result->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 2.0));
            return stat;
        }
        else if (x >= 0.5)
        {
            return lngamma_lanczos(x, result);
        }
        else if (x == 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (fabs(x) < 0.02)
        {
            double sgn;
            return lngamma_sgn_0(x, result, &sgn);
        }
        else if (x > -0.5 / (GSL_DBL_EPSILON * M_PI))
        {
            /* Try to extract a fractional
             * part from x.
             */
            double z = 1.0 - x;
            double s = sin(M_PI * z);
            double as = fabs(s);
            if (s == 0.0)
            {
                DOMAIN_ERROR(result);
            }
            else if (as < M_PI * 0.015)
            {
                /* x is near a negative integer, -N */
                if (x < INT_MIN + 2.0)
                {
                    result->val = 0.0;
                    result->err = 0.0;
                    GSL_ERROR("error", GSL_EROUND);
                }
                else
                {
                    int N = -(int)(x - 0.5);
                    double eps = x + N;
                    double sgn;
                    return lngamma_sgn_sing(N, eps, result, &sgn);
                }
            }
            else
            {
                gsl_sf_result lg_z;
                lngamma_lanczos(z, &lg_z);
                result->val = M_LNPI - (log(as) + lg_z.val);
                result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val) + lg_z.err;
                return GSL_SUCCESS;
            }
        }
        else
        {
            /* |x| was too large to extract any fractional part */
            result->val = 0.0;
            result->err = 0.0;
            GSL_ERROR("error", GSL_EROUND);
        }
    }

    int gsl_sf_lngamma_sgn_e(double x, gsl_sf_result *result_lg, double *sgn)
    {
        if (fabs(x - 1.0) < 0.01)
        {
            int stat = lngamma_1_pade(x - 1.0, result_lg);
            result_lg->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 1.0));
            *sgn = 1.0;
            return stat;
        }
        else if (fabs(x - 2.0) < 0.01)
        {
            int stat = lngamma_2_pade(x - 2.0, result_lg);
            result_lg->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 2.0));
            *sgn = 1.0;
            return stat;
        }
        else if (x >= 0.5)
        {
            *sgn = 1.0;
            return lngamma_lanczos(x, result_lg);
        }
        else if (x == 0.0)
        {
            *sgn = 0.0;
            DOMAIN_ERROR(result_lg);
        }
        else if (fabs(x) < 0.02)
        {
            return lngamma_sgn_0(x, result_lg, sgn);
        }
        else if (x > -0.5 / (GSL_DBL_EPSILON * M_PI))
        {
            /* Try to extract a fractional
             * part from x.
             */
            double z = 1.0 - x;
            double s = sin(M_PI * x);
            double as = fabs(s);
            if (s == 0.0)
            {
                *sgn = 0.0;
                DOMAIN_ERROR(result_lg);
            }
            else if (as < M_PI * 0.015)
            {
                /* x is near a negative integer, -N */
                if (x < INT_MIN + 2.0)
                {
                    result_lg->val = 0.0;
                    result_lg->err = 0.0;
                    *sgn = 0.0;
                    GSL_ERROR("error", GSL_EROUND);
                }
                else
                {
                    int N = -(int)(x - 0.5);
                    double eps = x + N;
                    return lngamma_sgn_sing(N, eps, result_lg, sgn);
                }
            }
            else
            {
                gsl_sf_result lg_z;
                lngamma_lanczos(z, &lg_z);
                *sgn = (s > 0.0 ? 1.0 : -1.0);
                result_lg->val = M_LNPI - (log(as) + lg_z.val);
                result_lg->err = 2.0 * GSL_DBL_EPSILON * fabs(result_lg->val) + lg_z.err;
                return GSL_SUCCESS;
            }
        }
        else
        {
            /* |x| was too large to extract any fractional part */
            result_lg->val = 0.0;
            result_lg->err = 0.0;
            *sgn = 0.0;
            GSL_ERROR("x too large to extract fraction part", GSL_EROUND);
        }
    }

    /****************************************/
    /* gamma_inc.c */
    /****************************************/

    /* Continued fraction which occurs in evaluation
     * of Q(a,x) or Gamma(a,x).
     *
     *              1   (1-a)/x  1/x  (2-a)/x   2/x  (3-a)/x
     *   F(a,x) =  ---- ------- ----- -------- ----- -------- ...
     *             1 +   1 +     1 +   1 +      1 +   1 +
     *
     * Hans E. Plesser, 2002-01-22 (hans dot plesser at itf dot nlh dot no).
     *
     * Split out from gamma_inc_Q_CF() by GJ [Tue Apr  1 13:16:41 MST 2003].
     * See gamma_inc_Q_CF() below.
     *
     */
    int gamma_inc_F_CF(const double a, const double x, gsl_sf_result *result)
    {
        const int nmax = 5000;
        const double small = gsl_pow_3(GSL_DBL_EPSILON);

        double hn = 1.0; /* convergent */
        double Cn = 1.0 / small;
        double Dn = 1.0;
        int n;

        /* n == 1 has a_1, b_1, b_0 independent of a,x,
           so that has been done by hand                */
        for (n = 2; n < nmax; n++)
        {
            double an;
            double delta;

            if (GSL_IS_ODD(n))
                an = 0.5 * (n - 1) / x;
            else
                an = (0.5 * n - a) / x;

            Dn = 1.0 + an * Dn;
            if (fabs(Dn) < small)
                Dn = small;
            Cn = 1.0 + an / Cn;
            if (fabs(Cn) < small)
                Cn = small;
            Dn = 1.0 / Dn;
            delta = Cn * Dn;
            hn *= delta;
            if (fabs(delta - 1.0) < GSL_DBL_EPSILON)
                break;
        }

        result->val = hn;
        result->err = 2.0 * GSL_DBL_EPSILON * fabs(hn);
        result->err += GSL_DBL_EPSILON * (2.0 + 0.5 * n) * fabs(result->val);

        if (n == nmax)
            GSL_ERROR("error in CF for F(a,x)", GSL_EMAXITER);
        else
            return GSL_SUCCESS;
    }

    /* Q large x asymptotic
     */
    int gamma_inc_Q_large_x(const double a, const double x, gsl_sf_result *result)
    {
        const int nmax = 5000;

        gsl_sf_result D;
        const int stat_D = gamma_inc_D(a, x, &D);

        double sum = 1.0;
        double term = 1.0;
        double last = 1.0;
        int n;
        for (n = 1; n < nmax; n++)
        {
            term *= (a - n) / x;
            if (fabs(term / last) > 1.0)
                break;
            if (fabs(term / sum) < GSL_DBL_EPSILON)
                break;
            sum += term;
            last = term;
        }

        result->val = D.val * (a / x) * sum;
        result->err = D.err * fabs((a / x) * sum);
        result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

        if (n == nmax)
            GSL_ERROR("error in large x asymptotic", GSL_EMAXITER);
        else
            return stat_D;
    }

    /* Continued fraction for Q.
     *
     * Q(a,x) = D(a,x) a/x F(a,x)
     *
     * Hans E. Plesser, 2002-01-22 (hans dot plesser at itf dot nlh dot no):
     *
     * Since the Gautschi equivalent series method for CF evaluation may lead
     * to singularities, I have replaced it with the modified Lentz algorithm
     * given in
     *
     * I J Thompson and A R Barnett
     * Coulomb and Bessel Functions of Complex Arguments and Order
     * J Computational Physics 64:490-509 (1986)
     *
     * In consequence, gamma_inc_Q_CF_protected() is now obsolete and has been
     * removed.
     *
     * Identification of terms between the above equation for F(a, x) and
     * the first equation in the appendix of Thompson&Barnett is as follows:
     *
     *    b_0 = 0, b_n = 1 for all n > 0
     *
     *    a_1 = 1
     *    a_n = (n/2-a)/x    for n even
     *    a_n = (n-1)/(2x)   for n odd
     *
     */
    int gamma_inc_Q_CF(const double a, const double x, gsl_sf_result *result)
    {
        gsl_sf_result D;
        gsl_sf_result F;
        const int stat_D = gamma_inc_D(a, x, &D);
        const int stat_F = gamma_inc_F_CF(a, x, &F);

        result->val = D.val * (a / x) * F.val;
        result->err = D.err * fabs((a / x) * F.val) + fabs(D.val * a / x * F.err);

        return GSL_ERROR_SELECT_2(stat_F, stat_D);
    }

    /* The dominant part,
     * D(a,x) := x^a e^(-x) / Gamma(a+1)
     */
    int gamma_inc_D(const double a, const double x, gsl_sf_result *result)
    {
        if (a < 10.0)
        {
            double lnr;
            gsl_sf_result lg;
            gsl_sf_lngamma_e(a + 1.0, &lg);
            lnr = a * log(x) - x - lg.val;
            result->val = exp(lnr);
            result->err = 2.0 * GSL_DBL_EPSILON * (fabs(lnr) + 1.0) * fabs(result->val);
            return GSL_SUCCESS;
        }
        else
        {
            gsl_sf_result gstar;
            gsl_sf_result ln_term;
            double term1;
            if (x < 0.5 * a)
            {
                double u = x / a;
                double ln_u = log(u);
                ln_term.val = ln_u - u + 1.0;
                ln_term.err = (fabs(ln_u) + fabs(u) + 1.0) * GSL_DBL_EPSILON;
            }
            else
            {
                double mu = (x - a) / a;
                gsl_sf_log_1plusx_mx_e(mu, &ln_term); /* log(1+mu) - mu */
                /* Propagate cancellation error from x-a, since the absolute
                   error of mu=x-a is DBL_EPSILON */
                ln_term.err += GSL_DBL_EPSILON * fabs(mu);
            };
            gsl_sf_gammastar_e(a, &gstar);
            term1 = exp(a * ln_term.val) / sqrt(2.0 * M_PI * a);
            result->val = term1 / gstar.val;
            result->err = 2.0 * GSL_DBL_EPSILON * (fabs(a * ln_term.val) + 1.0) * fabs(result->val);
            /* Include propagated error from log term */
            result->err += fabs(a) * ln_term.err * fabs(result->val);
            result->err += gstar.err / fabs(gstar.val) * fabs(result->val);
            return GSL_SUCCESS;
        }
    }

    /* P series representation.
     */
    int gamma_inc_P_series(const double a, const double x, gsl_sf_result *result)
    {
        const int nmax = 10000;

        gsl_sf_result D;
        int stat_D = gamma_inc_D(a, x, &D);

        /* Approximating the terms of the series using Stirling's
           approximation gives t_n = (x/a)^n * exp(-n(n+1)/(2a)), so the
           convergence condition is n^2 / (2a) + (1-(x/a) + (1/2a)) n >>
           -log(GSL_DBL_EPS) if we want t_n < O(1e-16) t_0. The condition
           below detects cases where the minimum value of n is > 5000 */

        if (x > 0.995 * a && a > 1e5)
        { /* Difficult case: try continued fraction */
            gsl_sf_result cf_res;
            int status = gsl_sf_exprel_n_CF_e(a, x, &cf_res);
            result->val = D.val * cf_res.val;
            result->err = fabs(D.val * cf_res.err) + fabs(D.err * cf_res.val);
            return status;
        }

        /* Series would require excessive number of terms */

        if (x > (a + nmax))
        {
            GSL_ERROR("gamma_inc_P_series x>>a exceeds range", GSL_EMAXITER);
        }

        /* Normal case: sum the series */

        {
            double sum = 1.0;
            double term = 1.0;
            double remainder;
            int n;

            /* Handle lower part of the series where t_n is increasing, |x| > a+n */

            int nlow = (x > a) ? (x - a) : 0;

            for (n = 1; n < nlow; n++)
            {
                term *= x / (a + n);
                sum += term;
            }

            /* Handle upper part of the series where t_n is decreasing, |x| < a+n */

            for (/* n = previous n */; n < nmax; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (fabs(term / sum) < GSL_DBL_EPSILON)
                    break;
            }

            /*  Estimate remainder of series ~ t_(n+1)/(1-x/(a+n+1)) */
            {
                double tnp1 = (x / (a + n)) * term;
                remainder = tnp1 / (1.0 - x / (a + n + 1.0));
            }

            result->val = D.val * sum;
            result->err = D.err * fabs(sum) + fabs(D.val * remainder);
            result->err += (1.0 + n) * GSL_DBL_EPSILON * fabs(result->val);

            if (n == nmax && fabs(remainder / sum) > GSL_SQRT_DBL_EPSILON)
                GSL_ERROR("gamma_inc_P_series failed to converge", GSL_EMAXITER);
            else
                return stat_D;
        }
    }

    /* Useful for small a and x. Handles the subtraction analytically.
     */
    int gamma_inc_Q_series(const double a, const double x, gsl_sf_result *result)
    {
        double term1; /* 1 - x^a/Gamma(a+1) */
        double sum;   /* 1 + (a+1)/(a+2)(-x)/2! + (a+1)/(a+3)(-x)^2/3! + ... */
        int stat_sum;
        double term2; /* a temporary variable used at the end */

        {
            /* Evaluate series for 1 - x^a/Gamma(a+1), small a
             */
            const double pg21 = -2.404113806319188570799476; /* PolyGamma[2,1] */
            const double lnx = log(x);
            const double el = M_EULER + lnx;
            const double c1 = -el;
            const double c2 = M_PI * M_PI / 12.0 - 0.5 * el * el;
            const double c3 = el * (M_PI * M_PI / 12.0 - el * el / 6.0) + pg21 / 6.0;
            const double c4 = -0.04166666666666666667 * (-1.758243446661483480 + lnx) * (-0.764428657272716373 + lnx) * (0.723980571623507657 + lnx) * (4.107554191916823640 + lnx);
            const double c5 = -0.0083333333333333333 * (-2.06563396085715900 + lnx) * (-1.28459889470864700 + lnx) * (-0.27583535756454143 + lnx) * (1.33677371336239618 + lnx) * (5.17537282427561550 + lnx);
            const double c6 = -0.0013888888888888889 * (-2.30814336454783200 + lnx) * (-1.65846557706987300 + lnx) * (-0.88768082560020400 + lnx) * (0.17043847751371778 + lnx) * (1.92135970115863890 + lnx) * (6.22578557795474900 + lnx);
            const double c7 = -0.00019841269841269841 * (-2.5078657901291800 + lnx) * (-1.9478900888958200 + lnx) * (-1.3194837322612730 + lnx) * (-0.5281322700249279 + lnx) * (0.5913834939078759 + lnx) * (2.4876819633378140 + lnx) * (7.2648160783762400 + lnx);
            const double c8 = -0.00002480158730158730 * (-2.677341544966400 + lnx) * (-2.182810448271700 + lnx) * (-1.649350342277400 + lnx) * (-1.014099048290790 + lnx) * (-0.191366955370652 + lnx) * (0.995403817918724 + lnx) * (3.041323283529310 + lnx) * (8.295966556941250 + lnx);
            const double c9 = -2.75573192239859e-6 * (-2.8243487670469080 + lnx) * (-2.3798494322701120 + lnx) * (-1.9143674728689960 + lnx) * (-1.3814529102920370 + lnx) * (-0.7294312810261694 + lnx) * (0.1299079285269565 + lnx) * (1.3873333251885240 + lnx) * (3.5857258865210760 + lnx) * (9.3214237073814600 + lnx);
            const double c10 = -2.75573192239859e-7 * (-2.9540329644556910 + lnx) * (-2.5491366926991850 + lnx) * (-2.1348279229279880 + lnx) * (-1.6741881076349450 + lnx) * (-1.1325949616098420 + lnx) * (-0.4590034650618494 + lnx) * (0.4399352987435699 + lnx) * (1.7702236517651670 + lnx) * (4.1231539047474080 + lnx) * (10.342627908148680 + lnx);

            term1 = a * (c1 + a * (c2 + a * (c3 + a * (c4 + a * (c5 + a * (c6 + a * (c7 + a * (c8 + a * (c9 + a * c10)))))))));
        }

        {
            /* Evaluate the sum.
             */
            const int nmax = 5000;
            double t = 1.0;
            int n;
            sum = 1.0;

            for (n = 1; n < nmax; n++)
            {
                t *= -x / (n + 1.0);
                sum += (a + 1.0) / (a + n + 1.0) * t;
                if (fabs(t / sum) < GSL_DBL_EPSILON)
                    break;
            }

            if (n == nmax)
                stat_sum = GSL_EMAXITER;
            else
                stat_sum = GSL_SUCCESS;
        }

        term2 = (1.0 - term1) * a / (a + 1.0) * x * sum;
        result->val = term1 + term2;
        result->err = GSL_DBL_EPSILON * (fabs(term1) + 2.0 * fabs(term2));
        result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
        return stat_sum;
    }

    int gsl_sf_gamma_inc_P_e(const double a, const double x, gsl_sf_result *result)
    {
        if (a <= 0.0 || x < 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (x == 0.0)
        {
            result->val = 0.0;
            result->err = 0.0;
            return GSL_SUCCESS;
        }
        else if (x < 20.0 || x < 0.5 * a)
        {
            /* Do the easy series cases. Robust and quick.
             */
            return gamma_inc_P_series(a, x, result);
        }
        else if (a > 1.0e+06 && (x - a) * (x - a) < a)
        {
            /* Crossover region. Note that Q and P are
             * roughly the same order of magnitude here,
             * so the subtraction is stable.
             */
            gsl_sf_result Q;
            int stat_Q = gamma_inc_Q_asymp_unif(a, x, &Q);
            result->val = 1.0 - Q.val;
            result->err = Q.err;
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return stat_Q;
        }
        else if (a <= x)
        {
            /* Q <~ P in this area, so the
             * subtractions are stable.
             */
            gsl_sf_result Q;
            int stat_Q;
            if (a > 0.2 * x)
            {
                stat_Q = gamma_inc_Q_CF(a, x, &Q);
            }
            else
            {
                stat_Q = gamma_inc_Q_large_x(a, x, &Q);
            }
            result->val = 1.0 - Q.val;
            result->err = Q.err;
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return stat_Q;
        }
        else
        {
            if ((x - a) * (x - a) < a)
            {
                /* This condition is meant to insure
                 * that Q is not very close to 1,
                 * so the subtraction is stable.
                 */
                gsl_sf_result Q;
                int stat_Q = gamma_inc_Q_CF(a, x, &Q);
                result->val = 1.0 - Q.val;
                result->err = Q.err;
                result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
                return stat_Q;
            }
            else
            {
                return gamma_inc_P_series(a, x, result);
            }
        }
    }

    /* Uniform asymptotic for x near a, a and x large.
     * See [Temme, p. 285]
     */
    int gamma_inc_Q_asymp_unif(const double a, const double x, gsl_sf_result *result)
    {
        const double rta = sqrt(a);
        const double eps = (x - a) / a;

        gsl_sf_result ln_term;
        const int stat_ln = gsl_sf_log_1plusx_mx_e(eps, &ln_term); /* log(1+eps) - eps */
        const double eta = GSL_SIGN(eps) * sqrt(-2.0 * ln_term.val);

        gsl_sf_result erfc;

        double R;
        double c0, c1;

        /* This used to say erfc(eta*M_SQRT2*rta), which is wrong.
         * The sqrt(2) is in the denominator. Oops.
         * Fixed: [GJ] Mon Nov 15 13:25:32 MST 2004
         */
        gsl_sf_erfc_e(eta * rta / M_SQRT2, &erfc);

        if (fabs(eps) < GSL_ROOT5_DBL_EPSILON)
        {
            c0 = -1.0 / 3.0 + eps * (1.0 / 12.0 - eps * (23.0 / 540.0 - eps * (353.0 / 12960.0 - eps * 589.0 / 30240.0)));
            c1 = -1.0 / 540.0 - eps / 288.0;
        }
        else
        {
            const double rt_term = sqrt(-2.0 * ln_term.val / (eps * eps));
            const double lam = x / a;
            c0 = (1.0 - 1.0 / rt_term) / eps;
            c1 = -(eta * eta * eta * (lam * lam + 10.0 * lam + 1.0) - 12.0 * eps * eps * eps) / (12.0 * eta * eta * eta * eps * eps * eps);
        }

        R = exp(-0.5 * a * eta * eta) / (M_SQRT2 * M_SQRTPI * rta) * (c0 + c1 / a);

        result->val = 0.5 * erfc.val + R;
        result->err = GSL_DBL_EPSILON * fabs(R * 0.5 * a * eta * eta) + 0.5 * erfc.err;
        result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

        return stat_ln;
    }

    /*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

    int gsl_sf_gamma_inc_Q_e(const double a, const double x, gsl_sf_result *result)
    {
        if (a < 0.0 || x < 0.0)
        {
            DOMAIN_ERROR(result);
        }
        else if (x == 0.0)
        {
            result->val = 1.0;
            result->err = 0.0;
            return GSL_SUCCESS;
        }
        else if (a == 0.0)
        {
            result->val = 0.0;
            result->err = 0.0;
            return GSL_SUCCESS;
        }
        else if (x <= 0.5 * a)
        {
            /* If the series is quick, do that. It is
             * robust and simple.
             */
            gsl_sf_result P;
            int stat_P = gamma_inc_P_series(a, x, &P);
            result->val = 1.0 - P.val;
            result->err = P.err;
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return stat_P;
        }
        else if (a >= 1.0e+06 && (x - a) * (x - a) < a)
        {
            /* Then try the difficult asymptotic regime.
             * This is the only way to do this region.
             */
            return gamma_inc_Q_asymp_unif(a, x, result);
        }
        else if (a < 0.2 && x < 5.0)
        {
            /* Cancellations at small a must be handled
             * analytically; x should not be too big
             * either since the series terms grow
             * with x and log(x).
             */
            return gamma_inc_Q_series(a, x, result);
        }
        else if (a <= x)
        {
            if (x <= 1.0e+06)
            {
                /* Continued fraction is excellent for x >~ a.
                 * We do not let x be too large when x > a since
                 * it is somewhat pointless to try this there;
                 * the function is rapidly decreasing for
                 * x large and x > a, and it will just
                 * underflow in that region anyway. We
                 * catch that case in the standard
                 * large-x method.
                 */
                return gamma_inc_Q_CF(a, x, result);
            }
            else
            {
                return gamma_inc_Q_large_x(a, x, result);
            }
        }
        else
        {
            if (x > a - sqrt(a))
            {
                /* Continued fraction again. The convergence
                 * is a little slower here, but that is fine.
                 * We have to trade that off against the slow
                 * convergence of the series, which is the
                 * only other option.
                 */
                return gamma_inc_Q_CF(a, x, result);
            }
            else
            {
                gsl_sf_result P;
                int stat_P = gamma_inc_P_series(a, x, &P);
                result->val = 1.0 - P.val;
                result->err = P.err;
                result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
                return stat_P;
            }
        }
    }

    /*-*-*-*-*-*-*-*-*-* Functions w/ Natural Prototypes *-*-*-*-*-*-*-*-*-*-*/

    double gsl_sf_gamma_inc_P(const double a, const double x)
    {
        EVAL_RESULT(gsl_sf_gamma_inc_P_e(a, x, &result));
    }

    double gsl_sf_gamma_inc_Q(const double a, const double x)
    {
        EVAL_RESULT(gsl_sf_gamma_inc_Q_e(a, x, &result));
    }

    /****************************************/
    /* beta_inc.c */
    /****************************************/

    double beta_cont_frac(const double a, const double b, const double x,
                          const double epsabs)
    {
        const unsigned int max_iter = 512;       /* control iterations      */
        const double cutoff = 2.0 * GSL_DBL_MIN; /* control the zero cutoff */
        unsigned int iter_count = 0;
        double cf;

        /* standard initialization for continued fraction */
        double num_term = 1.0;
        double den_term = 1.0 - (a + b) * x / (a + 1.0);

        if (fabs(den_term) < cutoff)
            den_term = GSL_NAN;

        den_term = 1.0 / den_term;
        cf = den_term;

        while (iter_count < max_iter)
        {
            const int k = iter_count + 1;
            double coeff = k * (b - k) * x / (((a - 1.0) + 2 * k) * (a + 2 * k));
            double delta_frac;

            /* first step */
            den_term = 1.0 + coeff * den_term;
            num_term = 1.0 + coeff / num_term;

            if (fabs(den_term) < cutoff)
                den_term = GSL_NAN;

            if (fabs(num_term) < cutoff)
                num_term = GSL_NAN;

            den_term = 1.0 / den_term;

            delta_frac = den_term * num_term;
            cf *= delta_frac;

            coeff = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1.0));

            /* second step */
            den_term = 1.0 + coeff * den_term;
            num_term = 1.0 + coeff / num_term;

            if (fabs(den_term) < cutoff)
                den_term = GSL_NAN;

            if (fabs(num_term) < cutoff)
                num_term = GSL_NAN;

            den_term = 1.0 / den_term;

            delta_frac = den_term * num_term;
            cf *= delta_frac;

            if (fabs(delta_frac - 1.0) < 2.0 * GSL_DBL_EPSILON)
                break;

            if (cf * fabs(delta_frac - 1.0) < epsabs)
                break;

            ++iter_count;
        }

        if (iter_count >= max_iter)
            return GSL_NAN;

        return cf;
    }

    /* The function beta_inc_AXPY(A,Y,a,b,x) computes A * beta_inc(a,b,x)
       + Y taking account of possible cancellations when using the
       hypergeometric transformation beta_inc(a,b,x)=1-beta_inc(b,a,1-x).

       It also adjusts the accuracy of beta_inc() to fit the overall
       absolute error when A*beta_inc is added to Y. (e.g. if Y >>
       A*beta_inc then the accuracy of beta_inc can be reduced) */

    double beta_inc_AXPY(const double A, const double Y,
                         const double a, const double b, const double x)
    {
        if (x == 0.0)
        {
            return A * 0 + Y;
        }
        else if (x == 1.0)
        {
            return A * 1 + Y;
        }
        else if (a > 1e5 && b < 10 && x > a / (a + b))
        {
            /* Handle asymptotic regime, large a, small b, x > peak [AS 26.5.17] */
            double N = a + (b - 1.0) / 2.0;
            return A * gsl_sf_gamma_inc_Q(b, -N * log(x)) + Y;
        }
        else if (b > 1e5 && a < 10 && x < b / (a + b))
        {
            /* Handle asymptotic regime, small a, large b, x < peak [AS 26.5.17] */
            double N = b + (a - 1.0) / 2.0;
            return A * gsl_sf_gamma_inc_P(a, -N * log1p(-x)) + Y;
        }
        else
        {
            double ln_beta = gsl_sf_lnbeta(a, b);
            double ln_pre = -ln_beta + a * log(x) + b * log1p(-x);

            double prefactor = exp(ln_pre);

            if (x < (a + 1.0) / (a + b + 2.0))
            {
                /* Apply continued fraction directly. */
                double epsabs = fabs(Y / (A * prefactor / a)) * GSL_DBL_EPSILON;

                double cf = beta_cont_frac(a, b, x, epsabs);

                return A * (prefactor * cf / a) + Y;
            }
            else
            {
                /* Apply continued fraction after hypergeometric transformation. */
                double epsabs =
                    fabs((A + Y) / (A * prefactor / b)) * GSL_DBL_EPSILON;
                double cf = beta_cont_frac(b, a, 1.0 - x, epsabs);
                double term = prefactor * cf / b;

                if (A == -Y)
                {
                    return -A * term;
                }
                else
                {
                    return A * (1 - term) + Y;
                }
            }
        }
    }

    /****************************************/
    /* beta.c */
    /****************************************/

    double isnegint(const double x)
    {
        return (x < 0) && (x == floor(x));
    }

    int gsl_sf_lnbeta_sgn_e(const double x, const double y, gsl_sf_result *result, double *sgn)
    {
        /* CHECK_POINTER(result) */

        if (x == 0.0 || y == 0.0)
        {
            *sgn = 0.0;
            DOMAIN_ERROR(result);
        }
        else if (isnegint(x) || isnegint(y))
        {
            *sgn = 0.0;
            DOMAIN_ERROR(result); /* not defined for negative integers */
        }

        /* See if we can handle the postive case with min/max < 0.2 */

        if (x > 0 && y > 0)
        {
            const double max = GSL_MAX(x, y);
            const double min = GSL_MIN(x, y);
            const double rat = min / max;

            if (rat < 0.2)
            {
                /* min << max, so be careful
                 * with the subtraction
                 */
                double lnpre_val;
                double lnpre_err;
                double lnpow_val;
                double lnpow_err;
                double t1, t2, t3;
                gsl_sf_result lnopr;
                gsl_sf_result gsx, gsy, gsxy;
                gsl_sf_gammastar_e(x, &gsx);
                gsl_sf_gammastar_e(y, &gsy);
                gsl_sf_gammastar_e(x + y, &gsxy);
                gsl_sf_log_1plusx_e(rat, &lnopr);
                lnpre_val = log(gsx.val * gsy.val / gsxy.val * M_SQRT2 * M_SQRTPI);
                lnpre_err = gsx.err / gsx.val + gsy.err / gsy.val + gsxy.err / gsxy.val;
                t1 = min * log(rat);
                t2 = 0.5 * log(min);
                t3 = (x + y - 0.5) * lnopr.val;
                lnpow_val = t1 - t2 - t3;
                lnpow_err = GSL_DBL_EPSILON * (fabs(t1) + fabs(t2) + fabs(t3));
                lnpow_err += fabs(x + y - 0.5) * lnopr.err;
                result->val = lnpre_val + lnpow_val;
                result->err = lnpre_err + lnpow_err;
                result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
                *sgn = 1.0;
                return GSL_SUCCESS;
            }
        }

        /* General case - Fallback */
        {
            gsl_sf_result lgx, lgy, lgxy;
            double sgx, sgy, sgxy, xy = x + y;
            int stat_gx = gsl_sf_lngamma_sgn_e(x, &lgx, &sgx);
            int stat_gy = gsl_sf_lngamma_sgn_e(y, &lgy, &sgy);
            int stat_gxy = gsl_sf_lngamma_sgn_e(xy, &lgxy, &sgxy);
            *sgn = sgx * sgy * sgxy;
            result->val = lgx.val + lgy.val - lgxy.val;
            result->err = lgx.err + lgy.err + lgxy.err;
            result->err += 2.0 * GSL_DBL_EPSILON * (fabs(lgx.val) + fabs(lgy.val) + fabs(lgxy.val));
            result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
            return GSL_ERROR_SELECT_3(stat_gx, stat_gy, stat_gxy);
        }
    }

    int gsl_sf_lnbeta_e(const double x, const double y, gsl_sf_result *result)
    {
        double sgn;
        int status = gsl_sf_lnbeta_sgn_e(x, y, result, &sgn);
        if (sgn == -1)
        {
            DOMAIN_ERROR(result);
        }
        return status;
    }

    double gsl_sf_lnbeta(const double x, const double y)
    {
        EVAL_RESULT(gsl_sf_lnbeta_e(x, y, &result));
    }

    double gsl_cdf_beta_P(const double &x, const double &a, const double &b)
    {
        double P;

        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        P = beta_inc_AXPY(1.0, 0.0, a, b, x);

        return P;
    }
}