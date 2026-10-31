package magicnet.bridge;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.IGameEntitiesFactory;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * Injection point for a custom controller.
 *
 * This mirrors forge.ai.LobbyPlayerAi exactly. Forge builds players through
 * IGameEntitiesFactory, so implementing this interface is the entire supported
 * mechanism for substituting a controller -- no engine patching, no log parsing.
 *
 * Wire it up with:  registeredPlayer.setPlayer(new InstrumentedLobbyPlayer(name))
 */
public class InstrumentedLobbyPlayer extends LobbyPlayer implements IGameEntitiesFactory {

    public InstrumentedLobbyPlayer(String name) {
        super(name);
    }

    private InstrumentedController controllerFor(Player p) {
        return new InstrumentedController(p.getGame(), p, this);
    }

    @Override
    public PlayerController createMindSlaveController(Player master, Player slave) {
        return controllerFor(slave);
    }

    @Override
    public Player createIngamePlayer(Game game, final int id) {
        Player p = new Player(getName(), game, id);
        p.setFirstController(controllerFor(p));
        return p;
    }

    @Override
    public void hear(LobbyPlayer player, String message) {
        // Not a chat participant.
    }
}
