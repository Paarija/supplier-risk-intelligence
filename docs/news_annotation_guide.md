# Disruption-news annotation guide

The benchmark is synthetic and intended for reproducible engineering demonstrations. It is
not evidence of real supplier events and must not be used for operational decisions.

## Labels

- `Port or transport disruption`: cargo, port, road, rail, border, or air-freight interruption.
- `Extreme weather`: weather or natural hazard that can affect production or logistics.
- `Factory shutdown`: a facility stops or materially reduces production.
- `Material shortage`: constrained availability of a required input or component.
- `Geopolitical disruption`: conflict, sanctions, export controls, tariffs, or political unrest.
- `Financial distress`: bankruptcy, insolvency, default, or loss of financing.
- `Quality or recall`: defective, contaminated, recalled, withdrawn, or quarantined output.
- `No disruption`: normal operations, cancelled threats, denials, or unrelated impact.

## Rules

1. Label the event described as currently affecting or credibly expected to affect supply.
2. Use `No disruption` when risk language is explicitly negated, cancelled, or out of scope.
3. Label the direct operational cause, not a secondary consequence. For example, a typhoon
   closing a port is `Extreme weather`, while an unrelated labor stoppage is transport risk.
4. If two events are present, use the event with the clearest direct operational impact.
5. Preserve hard negatives containing words such as “strike,” “shortage,” and “recall.”

## Quality process for a real dataset

Use two independent annotators, hide model predictions during labeling, measure Cohen's
kappa, adjudicate disagreements, and split related stories by event cluster so near-duplicate
coverage of one incident cannot appear in both training and test data.

