import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;

/**
 * Prints every Forge counter type as one JSON object per line, with the key
 * StateExtractor.addCounters emits for it ("ctr_" + lowercased getName()).
 * Run by scripts/gen_counter_keys.py; not part of the bridge.
 */
public class CounterKeys {
    public static void main(String[] args) {
        // CounterType.getValues() NPEs without a Lang instance (keyword names are localised).
        forge.util.Lang.createInstance("en-US");
        for (CounterType t : CounterType.getValues()) {
            String name = t.getName();
            StringBuilder sb = new StringBuilder("{");
            sb.append("\"key\":").append(q("ctr_" + name.toLowerCase()));
            sb.append(",\"name\":").append(q(name));
            sb.append(",\"display\":").append(q(t.getCounterOnCardDisplayName()));
            sb.append(",\"enum\":").append(t instanceof CounterEnumType ? q(((CounterEnumType) t).name()) : "null");
            sb.append(",\"keyword\":").append(t.isKeywordCounter());
            System.out.println(sb.append('}'));
        }
    }

    private static String q(String s) {
        if (s == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') {
                sb.append('\\').append(c);
            } else if (c < 0x20) {
                sb.append(String.format("\\u%04x", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
